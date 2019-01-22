using ApproxTools

using Test

rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tensor.jl")
include("bases/Monomials.jl")
include("LinearCombinations.jl")
