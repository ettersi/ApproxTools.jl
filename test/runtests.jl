using ApproxTools

using Test
using Compat

const BitsFloats = (Float32,Float64)
const Floats = (BitsFloats..., BigFloat)
const Reals = (Int, Floats...)
rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tensor.jl")
include("basis.jl")
include("barycentric.jl")
include("points.jl")
include("bernstein.jl")
include("fnorm.jl")
