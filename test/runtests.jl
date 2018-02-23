using ApproxTools
using Base.Test

const OldFloats = (Float32,Float64, Complex64, Complex128)
const Floats = (OldFloats..., BigFloat, Complex{BigFloat})

include("base.jl")
include("bary.jl")
include("chebyshev.jl")
