using ApproxTools
using Base.Test

const OldFloats = (Float32,Float64)
const Floats = (OldFloats..., BigFloat)

srand(42)

# Dummy rng for BigFloats
myrand(args...) = rand(args...)
myrand(::Type{BigFloat}, args...) = big.(rand(Float64,args...)) .+ eps(Float64)*big.(rand(Float64,args...))
myrand(::Type{Complex{BigFloat}}, args...) = big.(rand(Complex128,args...)) .+ eps(Float64)*big.(rand(Complex128,args...))

include("base.jl")
include("bary.jl")
include("chebyshev.jl")
include("utils.jl")
