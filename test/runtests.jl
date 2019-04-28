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
include("fnorm.jl")
include("bernstein.jl")
include("LinearCombinations.jl")
include("approximate.jl")
include("points/ChebyshevPoints.jl")
include("points/Midpoints.jl")
include("bases.jl")
include("LogNumbers.jl")
include("points/TrapezoidalPoints.jl")
