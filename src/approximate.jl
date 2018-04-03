abstract type ApproximationAlgorithm end

"""
    approximate([T,] f,n,alg::ApproximationAlgorithm)

Approximate `f` using algorithm `alg` with `n` degrees of freedom.

See `subtypes(ApproximationAlgorithm)` for a list of possible algorithm.
"""
function approximate end

approximate(
    f,
    n::Number,
    alg::ApproximationAlgorithm
) = approximate(f,(n,),alg)

approximate(
    T::Type,
    f,
    n::Number,
    alg::ApproximationAlgorithm
) = approximate(T,f,(n,),alg)

approximate(
    f,
    n::NTuple{<:Any,Integer},
    alg::ApproximationAlgorithm
) = approximate(Float64,f,n,alg)
