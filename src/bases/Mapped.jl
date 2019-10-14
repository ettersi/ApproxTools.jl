"""
    Mapped(B::AbstractBasis, x2x̂, x̂2x = nothing) = [ x-> B[k](x2x̂(x)) for k = 1:length(B)]

The inverse map `x̂2x` is only required for approximation but not for evaluation.
"""
struct Mapped{B,X2X̂,X̂2X} <: AbstractBasis
    basis::B
    x2x̂::X2X̂
    x̂2x::X̂2X
end
Mapped(b,x2x̂) = Mapped(b,x2x̂,nothing)

Base.length(B::Mapped) = length(B.basis)

approxpoints(B::Mapped{<:Any,<:Any,Nothing}) = ArgumentError("`approximate(f,Mapped(B,x2x̂))` is not defined if `x̂2x` is not provided")
approxpoints(B::Mapped) = B.x̂2x.(approxpoints(B.basis))
approxtransform(B::Mapped, f) = approxtransform(B.basis, f)

function iterate_basis(B::Mapped, x)
    Bx̂ = B.basis|B.x2x̂(x)
    b,state = IterTools.@ifsomething iterate(Bx̂)
    return b, (Bx̂,state)
end
function iterate_basis(B::Mapped, x, (Bx̂,state))
    b,state = IterTools.@ifsomething iterate(Bx̂,state)
    return b, (Bx̂,state)
end

evaluate_basis(B::Mapped,i,x) = B.basis[i](B.x2x̂(x))

evaltransform(B::Mapped, x, c) = evaltransform(B.basis,B.x2x̂.(x),c)
