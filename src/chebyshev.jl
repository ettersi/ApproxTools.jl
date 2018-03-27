"""
    chebpoints([T,] n)

The `n` Chebyshev points on [-1,1] in type `T`.
"""
chebpoints(n::Int) = chebpoints(Float64,n)
chebpoints(T::Type, n::Int) = Chebpoints{T}(n)
struct Chebpoints{T} <: AbstractVector{T}
    n::Int
    m::T
    b::T
    function Chebpoints{T}(n) where {T}
        if n == 1
            return new(n, T(0), T(π)/2)
        else
            return new(n, -T(π)/(n-1), T(π)*n/(n-1))
        end
    end
end
Base.size(x::Chebpoints) = (x.n,)
Base.getindex(x::Chebpoints, i::Int) = cos(x.m*i+x.b)


function baryweights(x::Chebpoints)
    T = eltype(x)
    n = length(x)
    return T(2),ChebBaryWeights{T}(n)
end
struct ChebBaryWeights{T} <: AbstractVector{T}
    n::Int
    f::T
    function ChebBaryWeights{T}(n) where {T}
        @assert n > 0
        if n > 1
            f = T(1)/(4*(n-1))
        else
            f = T(1)
        end
        new(n,f)
    end
end
Base.size(w::ChebBaryWeights) = (w.n,)
function Base.getindex(w::ChebBaryWeights, i::Int)
    n = w.n
    f = w.f
    T = eltype(w)
    r = f*ifelse(iseven(n-i),1,-1)
    if i == 1 || i == n
        r /= 2
    end
    return r
end
