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
    xx::Number,
    y = (),
    y2 = ()
)
    s,w = sw
    @assert length(x) == length(w) == length(f)
    n = length(x)

    I = 0
    l = one(float(promote_type(eltype.((s,xx,x,y,y2))...)))
    r = zero(float(promote_type(eltype.((w,f,xx,x))...)))
    @inbounds @simd for i = 1:n
        I = ifelse(xx == x[i], i, I)
        l *= s*(xx - x[i])
        r += w[i]*f[i]/(xx-x[i])
    end
    I != 0 && return typeof(l*r)(f[I])
    @inbounds @simd for i = 1:length(y)
        l /= xx - y[i]
    end
    @inbounds @simd for i = 1:length(y2)
        l /= xx^2 + y2[i]
    end
    return l*r
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
    xx::Number,
    y = (),
    y2 = ()
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
    if I != 0
        @. b = 0
        b[I] = 1
        return nothing
    end
    @inbounds @simd for i = 1:length(y)
        l /= xx - y[i]
    end
    @inbounds @simd for i = 1:length(y2)
        l /= xx^2 + y2[i]
    end
    @. b = l*w/(xx-x)
    return nothing
end

function bary(
    x::NTuple{N,AbstractVector},
    sw::NTuple{N,Tuple{Number,AbstractVector}},
    f::AbstractArray{<:Any,N},
    xx::NTuple{N,Union{Number,AbstractVector}},
    y = ntuple(i->(),Val(N)),
    y2 = ntuple(i->(),Val(N))
) where {N}
    return tucker(f, map(x,sw,xx,y,y2) do x,sw,xx,y,y2
        T = float(promote_type(eltype.((x,sw...,xx,y,y2))...))
        M = Matrix{T}(undef, length(x),length(xx))
        for j = 1:length(xx)
            baryvec!(view(M,:,j), x, sw, xx[j], y, y2)
        end
        return transpose(M)
    end)
end

struct BarycentricInterpolant{N,X,SW,F,Y,Y2}
    points::X
    scalingsandweights::SW
    values::F
    poles::Y
    cspoles::Y2
end
BarycentricInterpolant(
    x::NTuple{N,AbstractVector},
    sw::NTuple{N,Tuple{Number,AbstractVector}},
    f::AbstractArray{<:Any,N},
    y,y2
) where {N} = BarycentricInterpolant{N,typeof.((x,sw,f,y,y2))...}(x,sw,f,y,y2)

Base.ndims(::Type{BarycentricInterpolant{N,<:Any,<:Any,<:Any}}) where {N} = N
Base.ndims(::BarycentricInterpolant{N,<:Any,<:Any,<:Any}) where {N} = N

interpolate(
    x::AbstractVector,
    f::AbstractVector;
    poles = (),
    cspoles = ()
) = BarycentricInterpolant((x,), (baryweights(x,poles,cspoles),), f, (poles,), (cspoles,))
interpolate(
    x::NTuple{N,AbstractVector},
    f::AbstractArray{<:Any,N};
    poles = ntuple(i->(),Val(N)),
    cspoles = ntuple(i->(),Val(N))
) where {N} = BarycentricInterpolant(x, baryweights.(x,poles,cspoles), f, poles, cspoles)

(p::BarycentricInterpolant{1,<:Any,<:Any,<:Any,<:Any,<:Any})(xx::Number) = bary(p.points[1],p.scalingsandweights[1],p.values,xx,p.poles[1],p.cspoles[1])
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any,<:Any,<:Any})(xx::Vararg{Number,N}) where {N} = p(xx)
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any,<:Any,<:Any})(xx::NTuple{N,Number}) where {N} = first(bary(p.points,p.scalingsandweights,p.values,xx,p.poles,p.cspoles))
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any,<:Any,<:Any})(xx::NTuple{N,AbstractVector}) where {N} = bary(p.points,p.scalingsandweights,p.values,xx,p.poles,p.cspoles)
