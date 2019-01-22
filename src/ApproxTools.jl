module ApproxTools

using LinearAlgebra

include("MatFunUtils.jl")

include("tensor.jl")
export tucker, grideval

include("LinearCombinations.jl")
export LinearCombination, coeffs, basis

include("bases/Monomials.jl")
export Monomial

# include("core.jl")
# export interpolate, coeffs, basis, LinearCombination, Semiseparated, Monomial, Chebyshev, Weighted, Radial, Newton

# include("barycentric.jl")
# export Barycentric, prodpot, logpot
#
# include("points.jl")
# export chebpoints, equipoints, lejasort

include("fnorm.jl")
export fnorm

include("bernstein.jl")
export rsmo, jouk, ijouk, semimajor, semiminor, radius

end # module
