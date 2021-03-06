"""
    Chebyshev(n)

Basis of Chebyshev polynomials up to degree `n`.
"""
struct Chebyshev <: AbstractBasis
    n::Int
end

Base.length(B::Chebyshev) = B.n

approxpoints(B::Chebyshev) = ChebyshevPoints(B.n)

fftwtype(::Type{T}) where {T <: FFTW.fftwNumber} = T
fftwtype(::Type{T}) where {T <: Real} = Float64
fftwtype(::Type{T}) where {T <: Complex} = ComplexF64
function approxtransform(B::Chebyshev, f)
    n = length(B)
    T = fftwtype(eltype(f))
    n == 1 && return convert(Array{T},f)
    c = FFTW.r2r(f,FFTW.REDFT00,1)
    d = inv(real(T)(n-1)).*(i->isodd(i) ? 1 : -1).(1:n)
    d[1] /= 2; d[end] /= 2
    return d .* c
end

function iterate_basis(B::Chebyshev, x)
    length(B) == 0 && return nothing
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

# Use FFT to evaluate in Chebyshev points
function evaltransform(B::Chebyshev, x::ChebyshevPoints, c)
    # Warning: this code is not tested for mixed precision / BigFloat computations
    @assert length(B) == size(c,1)
    T = fftwtype(promote_type(eltype(x), eltype(c)))

    nB = length(B)-1
    nx = length(x)-1

    if nB == 0
        return one.(x) .* c
    elseif nx == 0
        return Matrix(B,x)*c
    else
        k = (nB+nx-1)÷nx
        pnB = k*nx
        O = zeros(eltype(c), (pnB - nB, Base.tail(size(c))...))
        d = inv(real(T)(2)).*(i->isodd(i) ? 1 : -1).(1:pnB+1); d[1] = 1; d[end] *= 2
        idx = ( 1:k:k*(nx+1), ntuple(i->1:size(c,i+1), ndims(c)-1)... )
        return FFTW.r2r(d.*[c;O], FFTW.REDFT00,1)[idx...]
    end
end
