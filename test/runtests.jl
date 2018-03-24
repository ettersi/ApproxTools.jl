using ApproxTools
if isdefined(Base, :Test) && !Base.isdeprecated(Base, :Test)
    using Base.Test
else
    using Test
end
using Compat

macro inferred07(expr)
    if VERSION < v"0.7-"
        return esc(expr)
    else
        return esc(:(@inferred($expr)))
    end
end

const BitsFloats = (Float32,Float64)
const Floats = (BitsFloats..., BigFloat)
const Reals = (Int, Floats...)
rnc(reals::Type) = (reals,complex.(reals))
rnc(reals::Tuple) = (reals...,complex.(reals)...)

include("tensor.jl")
include("interpolation.jl")
include("chebyshev.jl")
