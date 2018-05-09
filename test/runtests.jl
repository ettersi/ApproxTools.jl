using ApproxTools
if isdefined(Base, :Test) && !Base.isdeprecated(Base, :Test)
    using Base.Test
else
    using Test
end
using Compat

const BitsFloats = (Float32,Float64)
const Floats = (BitsFloats..., BigFloat)
const Reals = (Int, Floats...)
rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tensor.jl")
include("linearcombination.jl")
include("barycentric.jl")
include("chebyshev.jl")
