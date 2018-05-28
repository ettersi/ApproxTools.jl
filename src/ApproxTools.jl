module ApproxTools

using Compat

include("tensor.jl")
export tucker, cartesian, tensor, grideval

include("linearcombination.jl")
export interpolate

include("barycentric.jl")
export Barycentric

include("points.jl")
export chebpoints, equipoints

end # module
