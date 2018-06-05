module ApproxTools

using Compat

include("tensor.jl")
export cartesian, tensor, grideval

include("basis.jl")
export interpolate, coeffs, basis

include("barycentric.jl")
export Barycentric

include("points.jl")
export chebpoints, equipoints

end # module
