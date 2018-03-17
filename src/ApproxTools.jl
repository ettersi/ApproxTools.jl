module ApproxTools

include("base.jl");      export interpolate, map2refinterval
include("bary.jl");      export geometric_mean_distance,baryweights,bary,Barycentric
include("chebyshev.jl"); export chebpoints, chebcoeffs, chebeval
using Compat

end # module
