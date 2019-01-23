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



struct Chebyshev <: Basis
    n::Int
end

Base.length(b::Chebyshev) = b.n

evaluationpoints(b::Chebyshev) = chebpoints(b.n)

fftwtype(::Type{T}) where {T <: FFTW.fftwNumber} = T
fftwtype(::Type{T}) where {T <: Real} = Float64
fftwtype(::Type{T}) where {T <: Complex} = ComplexF64
approxtransform(b::Chebyshev) = f->begin
    n = length(b)
    T = fftwtype(eltype(f))
    n == 0 && return Array{T}(undef, size(f))
    n == 1 && return convert(Array{T},f)
    c = FFTW.r2r(f,FFTW.REDFT00,1)
    d = (real(T)(1)/(n-1)).*(i->isodd(i) ? 1 : -1).(1:n)
    d[1] /= 2; d[end] /= 2
    return Diagonal(d)*c
end

(b::Chebyshev)(x̂::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}) = ChebyshevValues(b,x̂)
(b::Chebyshev)(M::AbstractMatrix,v::AbstractVector) = b((M,v))

struct ChebyshevValues{X̂} <: BasisValues
    basis::Chebyshev
    evaluationpoint::X̂
end
Base.eltype(::Type{ChebyshevValues{X̂}}) where {X̂<:Number} = typeof(zero(X̂)*zero(X̂) + zero(X̂))
Base.eltype(::Type{ChebyshevValues{X̂}}) where {X̂<:Union{AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}} = MatFun.basis_eltype(X̂)

function Base.iterate(bv::ChebyshevValues)
    x̂ = bv.evaluationpoint
    T0 = MatFun.one(x̂)
    return T0,(2,T0,T0)
end
function Base.iterate(bv::ChebyshevValues, state)
    x̂ = bv.evaluationpoint
    i,T0,T1 = state
    i > length(bv) && return nothing
    if i == 2
        T0,T1 = T1, MatFun.xval(x̂)
    else
        T0,T1 = T1, 2*MatFun.xmul(x̂)*T1 - T0
    end
    return T1,(i+1,T0,T1)
end
