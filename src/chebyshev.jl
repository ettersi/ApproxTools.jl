"""
    chebpoints([T,] n)

The `n` Chebyshev points on [-1,1] in type `T`.
"""
chebpoints(T::Type, n::Int) = Chebpoints{T}(n)
chebpoints(n::Int) = chebpoints(Float64,n)
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

