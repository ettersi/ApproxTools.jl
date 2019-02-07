"""
    approximate(f, S) -> p::LinearCombination

Approximate `f` using the approximation scheme `S`.

Possible approximation schemes are:
 - Basis with associated canonical interpolation points (e.g. `S = Chebyshev(5)`).
 - Pair of interpolation point and basis (e.g. `S = ([-1,0,1], Monomials(3))`).
 - More to come.

# Examples
```
coeffs(approximate(x->x^2, Monomials(3))) ≈ [0,0,1]
coeffs(approximate(sin, ([-π,0,π],Monomials(3))) ≈ [0,0,0]
coeffs(approximate((x1,x2)->x1*x2, Monomials(2))) ≈ [0 0; 0 1]
```
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

evaluationpoints((x,B)::Tuple{AbstractVector,Basis}) = x
approxtransform((x,B)::Tuple{AbstractVector,Basis}) = f->Matrix(B,x)\f

basis(B::Basis) = B
basis((x,B)::Tuple{AbstractVector,Basis}) = B
