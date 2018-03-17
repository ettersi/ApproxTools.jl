module ApproxTools

using Compat

include("base.jl");            export interpolate
include("bary.jl");            export baryweights,bary,Barycentric
include("chebyshev.jl");       export chebpoints, chebcoeffs, chebeval
include("utils.jl");           export EmptyVector, map2refinterval

end # module
