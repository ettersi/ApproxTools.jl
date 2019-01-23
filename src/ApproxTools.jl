module ApproxTools

using LinearAlgebra
using FFTW

include("MatFun.jl")

include("tensor.jl")
export tucker, grideval

include("fnorm.jl")
export fnorm

include("bernstein.jl")
export rsmo, jouk, ijouk, semimajor, semiminor, radius

include("LinearCombinations.jl")
export LinearCombination, coeffs, basis

include("bases/Monomials.jl")
export Monomial

include("approximate.jl")
export approximate

include("bases/Chebyshev.jl")
export chebpoints, Chebyshev


end # module
