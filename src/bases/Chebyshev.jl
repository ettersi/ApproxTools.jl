"""
    Chebyshev(n)

Basis of Chebyshev polynomials up to degree `n`. 
"""
struct Chebyshev <: Basis
    n::Int
end

Base.length(B::Chebyshev) = B.n

evaluationpoints(B::Chebyshev) = chebpoints(B.n)

fftwtype(::Type{T}) where {T <: FFTW.fftwNumber} = T
fftwtype(::Type{T}) where {T <: Real} = Float64
fftwtype(::Type{T}) where {T <: Complex} = ComplexF64
approxtransform(B::Chebyshev) = f->begin
    n = length(B)
    T = fftwtype(eltype(f))
    n == 0 && return Array{T}(undef, size(f))
    n == 1 && return convert(Array{T},f)
    c = FFTW.r2r(f,FFTW.REDFT00,1)
    d = (real(T)(1)/(n-1)).*(i->isodd(i) ? 1 : -1).(1:n)
    d[1] /= 2; d[end] /= 2
    return Diagonal(d)*c
end

function iterate_basis(B::Chebyshev, x)
    T0 = one(x)
    return T0,(2,T0,T0)
end
function iterate_basis(B::Chebyshev, x, (i,T0,T1))
    i > length(B) && return nothing
    if i == 2
        T0,T1 = T1, val(x)
    else
        T0,T1 = T1, 2 * (x*T1) - T0
    end
    return T1,(i+1,T0,T1)
end
