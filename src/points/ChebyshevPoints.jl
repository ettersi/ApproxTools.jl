"""
    ChebyshevPoints{T = Float64}(n)

The `n` Chebyshev points on [-1,1] in type `T`.
"""
struct ChebyshevPoints{T <: AbstractFloat} <: AbstractVector{T}
    data::Vector{T}
    function ChebyshevPoints{T}(n::Integer) where {T}
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
ChebyshevPoints(n) = ChebyshevPoints{Float64}(n)
Base.size(x::ChebyshevPoints) = size(x.data)
Base.getindex(x::ChebyshevPoints, i::Int) = x.data[i]

function prodpot(x::ChebyshevPoints)
    T = eltype(x); n = length(x)
    n == 1 && return [LogNumber(1,float(T(0)))]
    p = LogNumber.( (-1).^(0:n-1), log(T(n-1)) - (n-2)*log(T(2)) )
    p[1] *= LogNumber(true,log(2)); p[end] *= LogNumber(true,log(2))
    return p
end
