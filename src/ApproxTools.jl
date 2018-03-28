module ApproxTools

using Compat

if VERSION < v"0.7-"
    # promote_type with four or more arguments is not type stable on 0.6.2
    Base.promote_type(A,B,C,D=Union{},E=Union{},F=Union{},G=Union{},H=Union{}) =
        promote_type(promote_type(promote_type(promote_type(promote_type(promote_type(promote_type(A,B),C),D),E),F),G),H)
end

include("tensor.jl")
export tucker, cartesian, tensor

include("interpolate.jl")
export baryweights, bary, interpolate

include("chebyshev.jl")
export chebpoints

end # module
