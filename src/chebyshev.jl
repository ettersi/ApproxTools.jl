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

function pointsprod(si,li,xi,x::Chebpoints,i)
    n = length(x)
    T = typeof(xi)
    if n != 1
        si *= ifelse(iseven(n-i),1,-1)
        li += (n-2)*log(T(2)) - log(T(ifelse(i == 1 || i == n, 2, 1)*(n-1)))
    end
    return si,li
end
