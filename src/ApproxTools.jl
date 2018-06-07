module ApproxTools

using Compat

include("basis.jl")
export interpolate, coeffs, basis, LinearCombination, Chebyshev, Weighted

include("tensor.jl")
export cartesian, tensor, grideval

include("barycentric.jl")
export Barycentric

include("points.jl")
export chebpoints, equipoints

end # module
