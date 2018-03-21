module ApproxTools

include("tensor.jl")
export tucker, cartesian, tensor

include("interpolation.jl")
export baryweights, bary, interpolate

end # module
