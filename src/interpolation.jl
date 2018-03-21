"""
    baryweights(x [,y [,y2]]) -> s,w

Compute the barycentric weights `s^n*w` for the `n` interpolation
points `x` and poles `y` and `±√(y2)*im`.

```
s^n *w[i] := ( prod(x[i] .- y) * prod(x[i]^2 .+ y2) ) / prod( x[i] .- x[!=i] )
```

The extra factor `s^n` allows to compute the barycentric weights
even in cases where `s^n*w` would over- or underflow.
"""
function baryweights(x, y = (), y2 = ())
    n = length(x)
    T = float(promote_type(eltype.((x,y,y2))...))

    s = ones(T,n)
    l = zeros(T,n)
    for i = 1:n
        xi = convert(promote_type(eltype(x),real(T)),x[i])
        # ^ Make sure we compute everything at highest precision
        @inbounds @simd for j = [1:i-1;i+1:n]
            s[i] *= conj(sign(xi - x[j]))
            l[i] -= log(abs(xi - x[j]))
        end
        @inbounds @simd for j = 1:length(y)
            s[i] *= sign(xi - y[j])
            l[i] += log(abs(xi - y[j]))
        end
        @inbounds @simd for j = 1:length(y2)
            s[i] *= sign(xi^2 + y2[j])
            l[i] += log(abs(xi^2 + y2[j]))
        end
    end

    # Scale weights to prevent over-/underflow
    ml = mean(l)
    return exp(ml/n), @. s*exp(l-ml)
end

function bary(
    x::AbstractVector,
    sw::Tuple{Number,AbstractVector},
    f::AbstractVector,
    xx::Number
)
    s,w = sw
    @assert length(x) == length(w) == length(f)
    n = length(x)

    I = 0
    l = one(float(promote_type(eltype.((s,xx,x))...)))
    r = zero(float(promote_type(eltype.((w,f,xx,x))...)))
    @inbounds @simd for i = 1:n
        I = ifelse(xx == x[i], i, I)
        l *= s*(xx - x[i])
        r += w[i]*f[i]/(xx-x[i])
    end
    return I == 0 ? l*r : typeof(l*r)(f[I])
end

"""
    baryvec!(b,s,x,xx)

Initialize `b` such that `sum(b.*w.*f)` evaluates the barycentric
interpolant to `(x,f)` with barycentric weights `w`.
"""
function baryvec!(
    b::AbstractVector,
    x::AbstractVector,
    sw::Tuple{Number,AbstractVector},
    xx::Number
)
    s,w = sw
    @assert length(b) == length(x)
    n = length(x)

    I = 0
    l = one(float(promote_type(eltype.((s,xx,x))...)))
    @inbounds @simd for i = 1:n
        I = ifelse(xx == x[i], i, I)
        l *= s*(xx - x[i])
    end
    if I == 0
        @. b = l*w/(xx-x)
    else
        @. b = 0
        b[I] = 1
    end
    return nothing
end

function bary(
    x::NTuple{N,AbstractVector},
    sw::NTuple{N,Tuple{Number,AbstractVector}},
    f::AbstractArray{<:Any,N},
    xx::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    return tucker(f, map(x,sw,xx) do x,sw,xx
        T = float(promote_type(eltype.((x,sw...,xx))...))
        M = Matrix{T}(undef, length(x),length(xx))
        for j = 1:length(xx)
            baryvec!(view(M,:,j), x, sw, xx[j])
        end
        return transpose(M)
    end)
end

struct BarycentricInterpolant{N,X,SW,F}
    points::X
    scalingsandweights::SW
    values::F
end

Base.ndims(::Type{BarycentricInterpolant{N,<:Any,<:Any,<:Any}}) where {N} = N
Base.ndims(::BarycentricInterpolant{N,<:Any,<:Any,<:Any}) where {N} = N

interpolate(
    x::AbstractVector,
    f::AbstractVector;
    kwargs...
) = interpolate((x,),f; kwargs...)
interpolate(
    x::NTuple{N,AbstractVector},
    f::AbstractArray{<:Any,N};
    poles = ntuple(i->(),Val(N)),
    cspoles = ntuple(i->(),Val(N))
) where {N} = interpolate(x, baryweights.(x, poles, cspoles), f)
interpolate(
    x::NTuple{N,AbstractVector},
    sw::NTuple{N,Tuple{Number,AbstractVector}},
    f::AbstractArray{<:Any,N}
) where {N} = BarycentricInterpolant{N,typeof(x),typeof(sw),typeof(f)}(x,sw,f)

(p::BarycentricInterpolant{1,<:Any,<:Any,<:Any})(xx::Number) = bary(p.points[1],p.scalingsandweights[1],p.values,xx)
(p::BarycentricInterpolant{1,<:Any,<:Any,<:Any})(xx...) = throw(MethodError(p,x))
# ^ Make sure we don't accidentally call the below method using e.g. p([0,1])
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx...) where {N} = p(xx)
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx::NTuple{N,Number}) where {N} = first(bary(p.points,p.scalingsandweights,p.values,xx))
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx::NTuple{N,AbstractVector}) where {N} = bary(p.points,p.scalingsandweights,p.values,xx)
