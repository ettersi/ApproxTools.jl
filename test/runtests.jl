using ApproxTools

using Test
using LinearAlgebra

const BitsFloats = (Float32,Float64)
const Floats = (BitsFloats..., BigFloat)
const Reals = (Int, Floats...)
rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tucker.jl")
include("grideval.jl")
include("LogNumbers.jl")
include("LinearCombinations.jl")
include("approximate.jl")
include("inner_products.jl")
include("points/ChebyshevPoints.jl")
include("points/Midpoints.jl")
include("points/TrapezoidalPoints.jl")
include("bases.jl")
