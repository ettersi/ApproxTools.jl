module ApproxTools

using Compat

include("tensor.jl")
export tucker, cartesian, tensor, grideval

include("linearcombination.jl")
export interpolate

include("barycentric.jl")
export Barycentric

include("chebyshev.jl")
export chebpoints

end # module
