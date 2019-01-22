using ApproxTools

using Test
using LinearAlgebra

rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tensor.jl")
include("bases/Monomials.jl")
include("LinearCombinations.jl")
include("fndims.jl)
include("fnorm.jl")
