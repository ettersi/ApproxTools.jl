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
            data = Vector{T}(undef,n)
            for i = 1:n
                data[i] = cos(T(π)*(n-i)/(n-1))
            end
            return new(data)
        end
    end
end
Base.size(x::Chebpoints) = size(x.data)
Base.getindex(x::Chebpoints, i::Int) = x.data[i]

function prodpot(x::Chebpoints)
    n = length(x)
    T = real(eltype(x))

    n == 1 && return [one(lognumber(T))]

    w = Vector{lognumber(T)}(undef,n)
    w[1] = LogNumber(ifelse(isodd(n),1,-1), log(T(n-1)) - (n-3)*log(T(2)))
    for i = 2:n-1
        w[i] = LogNumber(ifelse(isodd(n-i+1),1,-1), log(T(n-1)) - (n-2)*log(T(2)))
    end
    w[n] = LogNumber(1, log(T(n-1)) - (n-3)*log(T(2)))

    return w
end

