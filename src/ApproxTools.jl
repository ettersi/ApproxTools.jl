module ApproxTools

using LinearAlgebra
using SparseArrays
using FFTW

include("tensor.jl")
export tucker, grideval

include("fnorm.jl")
export fnorm

include("bernstein.jl")
export rsmo, jouk, ijouk, semimajor, semiminor, radius

include("ArgumentWrappers.jl")

include("LinearCombinations.jl")
export LinearCombination, coeffs, basis, @evaluate

include("approximate.jl")
export approximate

include("points/ChebyshevPoints.jl")
export ChebyshevPoints

include("points/Midpoints.jl")
export Midpoints

include("bases/Monomials.jl")
export Monomials

include("bases/Chebyshev.jl")
export Chebyshev

include("bases/Poles.jl")
export Poles

include("bases/Weighted.jl")
export Weighted

# include("bases/Barycentric.jl")
# export Barycentric, nodepoly

end # module
