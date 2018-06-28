module ApproxTools

using Compat

include("tensor.jl")
export cartesian, tensor, grideval

include("basis.jl")
export interpolate, coeffs, basis, LinearCombination, Semiseparated, Monomial, Chebyshev, Weighted

include("barycentric.jl")
export Barycentric, prodpot

include("points.jl")
export chebpoints, equipoints

include("bernstein.jl")
export rsmo, jouk, ijouk, semimajor, semiminor, radius

include("fnorm.jl")
export fnorm

end # module
