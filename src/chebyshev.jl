"""
    chebpoints([T,] n)

The `n` Chebyshev points on [-1,1] in type `T`.
"""
chebpoints(n::Int) = chebpoints(Float64,n)
chebpoints(T::Type{<:AbstractFloat}, n::Int) = Chebpoints{T}(n)
struct Chebpoints{T} <: AbstractVector{T}
    data::Vector{T}
    function Chebpoints{T}(n) where {T}
        if n == 1
            return new(T[0])
        else
            data = Vector{T}(n)
            for i = 1:n
                data[i] = cos(T(π)*(n-i)/(n-1))
            end
            return new(data)
        end
    end
end
Base.size(x::Chebpoints) = size(x.data)
Base.getindex(x::Chebpoints, i::Int) = x.data[i]


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
