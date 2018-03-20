module ApproxTools

include("tensor.jl")
export tucker, ×, ⊗

include("interpolation.jl")
export baryweights, bary, interpolate

end # module
