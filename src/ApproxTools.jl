module ApproxTools

using LinearAlgebra
using SparseArrays
using FFTW
using IterTools
using Statistics

include("tucker.jl")
export tucker

include("grideval.jl")
export grideval, @gridfun

include("LogNumbers.jl")

include("ArgumentWrappers.jl")

include("LinearCombinations.jl")
export LinearCombination, coeffs, basis, @evaluate

include("approximate.jl")
export approximate

include("inner_products.jl")
export WeightedL2, LegendreInnerProduct, JacobiInnerProduct, evaluate, evalpoints

include("points/ChebyshevPoints.jl")
export ChebyshevPoints

include("points/TrapezoidalPoints.jl")
export TrapezoidalPoints

include("points/Midpoints.jl")
export Midpoints

include("bases/Basis.jl")
export Basis

include("bases/Monomials.jl")
export Monomials

include("bases/NegativeMonomials.jl")
export NegativeMonomials

include("bases/Chebyshev.jl")
export Chebyshev

include("bases/Combined.jl")
export Combined

include("bases/Poles.jl")
export Poles

include("bases/Weighted.jl")
export Weighted

include("bases/Mapped.jl")
export Mapped

include("bases/Barycentric.jl")
export Barycentric

include("bases/LanczosBasis.jl")
export LanczosBasis

end # module
