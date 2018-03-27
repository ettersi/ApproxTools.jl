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
    @inbounds for i = 1:n
        xi = convert(promote_type(eltype(x),real(T)),x[i])
        # ^ Make sure we compute everything at highest precision

        # We factor the products into separate functions to allow
        # for dispatch to more efficient functions for particular
        # point sets (e.g. Chebyshev points).
        s[i],l[i] = pointsprod(s[i],l[i],xi,x,i)
        s[i],l[i] = poleprod(s[i],l[i],xi,y)
        s[i],l[i] = cspoleprod(s[i],l[i],xi,y2)
    end

    # Scale weights to prevent over-/underflow
    ml = mean(l)
    return exp(ml/n), @. s*exp(l-ml)
end

function pointsprod(si,li,xi,x,i)
    @inbounds @simd for j = [1:i-1;i+1:length(x)]
        si *= conj(sign(xi - x[j]))
        li -= log(abs(xi - x[j]))
    end
    return si,li
end
function poleprod(si,li,xi,y)
    @inbounds @simd for j = 1:length(y)
        si *= sign(xi - y[j])
        li += log(abs(xi - y[j]))
    end
    return si,li
end
function cspoleprod(si,li,xi,y2)
    @inbounds @simd for j = 1:length(y2)
        si *= sign(xi^2 + y2[j])
        li += log(abs(xi^2 + y2[j]))
    end
    return si,li
end


"""
    bary(x,sw,f,xx,y=(),y2=())

Evaluate the rational barycentric interpolation formula.

# Arguments
- `x`,`f`: Data to interpolate.
- `sw`: Barycentric scalings and weights. See [`baryweights`](@ref).
- `xx`: Evaluation point(s).
- `y`, `y2`: Simple and conjugate symmetric poles of the interpolant.

This function supports both one-dimensional interpolation as well as
interpolation on tensor product grids. Check the source code for
syntax details.
"""
function bary end

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

struct BarycentricInterpolant{X,SW,F,Y,Y2}
    points::X
    scalingsandweights::SW
    values::F
    poles::Y
    cspoles::Y2
end
const Bary{N} = BarycentricInterpolant{
    <: NTuple{N,AbstractVector},
    <: NTuple{N,Tuple{Number,AbstractVector}},
    <: AbstractArray{<:Any,N},
    <: NTuple{N,<:Any},
    <: NTuple{N,<:Any}
}

Base.ndims(::Type{<:Bary{N}}) where {N} = N
Base.ndims(::Bary{N}) where {N} = N


"""
    interpolate(x,f; poles=(), cspoles=()) -> r

Compute the rational function interpolating `(x,f)` with poles at `poles`
and `±√(cspoles)*im`.

This function supports both one-dimensional interpolation as well as
interpolation on tensor product grids.

# Examples
One dimension:
```
julia> p = interpolate([0,1],[0,1])
       p(0.5)
0.5
```

Two dimensions:
```
julia> x = ([0,1],[0,1])
       f = [0 0; 0 1]
       p = interpolate(x,f);

julia> p(0.5,0.5)
 0.25

julia> p([0,0.5,1],[0,0.5,1])
3×3 Array{Float64,2}:
 0.0  0.0   0.0
 0.0  0.25  0.5
 0.0  0.5   1.0
```
"""
function interpolate end

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

(p::Bary{N})(xx::Vararg{Union{Number,AbstractVector},N}) where {N} = p(xx)
(p::Bary{1})(xx::NTuple{1,Number}) = bary(p.points[1],p.scalingsandweights[1],p.values,xx[1],p.poles[1],p.cspoles[1])
(p::Bary{1})(xx::NTuple{1,AbstractVector}) = p.(xx[1])
(p::Bary{N})(xx::NTuple{N,Number}) where {N} = first(bary(p.points,p.scalingsandweights,p.values,xx,p.poles,p.cspoles))
(p::Bary{N})(xx::NTuple{N,Union{Number,AbstractVector}}) where {N} = bary(p.points,p.scalingsandweights,p.values,xx,p.poles,p.cspoles)
