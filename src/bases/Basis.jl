"""
    Basis(f)

Turn the vector of functions `f` into a basis.
"""
struct Basis{F} <: AbstractBasis
    functions::F
end

Base.length(B::Basis) = length(B.functions)

Base.getindex(B::Basis, i) = B.functions[i]
