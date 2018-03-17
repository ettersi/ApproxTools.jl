module ApproxTools

using Compat

include("utils.jl");           export EmptyArray, EmptyVector, EmptyMatrix, map2refinterval

include("base.jl");            export interpolate
include("bary.jl");            export baryweights, bary, Barycentric
include("chebyshev.jl");       export chebpoints, chebcoeffs, chebeval

end # module
