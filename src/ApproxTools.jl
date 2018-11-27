module ApproxTools

include("tensor.jl")
export tucker, grideval

include("basis.jl")
export interpolate, coeffs, basis, LinearCombination, Semiseparated, Monomial, Chebyshev, Weighted, Radial, Newton

include("barycentric.jl")
export Barycentric, prodpot, logpot

include("points.jl")
export chebpoints, equipoints, lejasort

include("bernstein.jl")
export rsmo, jouk, ijouk, ijoukt, semimajor, semiminor, radius, radiust

include("fnorm.jl")
export fnorm

end # module
