using ApproxTools

using Test
using LinearAlgebra

const BitsFloats = (Float32,Float64)
const Floats = (BitsFloats..., BigFloat)
const Reals = (Int, Floats...)
rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tensor.jl")
include("bases/Monomials.jl")
include("LinearCombinations.jl")
include("fndims.jl)
include("fnorm.jl")
include("bernstein.jl")
