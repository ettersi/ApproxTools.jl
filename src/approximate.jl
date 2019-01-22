"""
    approximate(f, S) -> p

Approximate `f` using the approximation scheme `S`.
"""
function approximate end

# Broadcast single argument to number of dimensions
approximate(f, S) = approximate(f, ntuple(i->S,Val(fndims(f))))
approximate(f, S::Tuple{AbstractVector,Basis}) = invoke(approximate, Tuple{Any,Any}, f,S)

# Generic algorithm for linear approximation schemes
approximate(f, S::NTuple{N,Any}) where {N} =
    LinearCombination(
        tucker(
            grideval(f,evaluationpoints.(S)),
            approxtransform.(S)
        ),
        basis.(S)
    )

evaluationpoints(xb::Tuple{AbstractVector,Basis}) = xb[1]
approxtransform(xb::Tuple{AbstractVector,Basis}) = f->begin
    x,b = xb
    collect(b,x)\f
end

basis(b::Basis) = b
basis(xb::Tuple{AbstractVector,Basis}) = xb[2]
