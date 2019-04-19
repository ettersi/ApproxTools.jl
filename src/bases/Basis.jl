"""
    Basis(f)

Turn the vector of functions `f` into a basis.
"""
struct Basis{F} <: AbstractBasis
    functions::F
end

Base.length(B::Basis) = length(B.functions)

evaluate_basis(B::Basis, i, x) = B.functions[i](x)
