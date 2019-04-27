"""
    Weighted(B::AbstractBasis, w) = [ x-> w(x) * B[k](x) for k = 1:length(B)]
"""
struct Weighted{B,W} <: AbstractBasis
    basis::B
    weight::W
end

Base.length(B::Weighted) = length(B.basis)

approxpoints(B::Weighted) = approxpoints(B.basis)
function approxtransform(B::Weighted, f)
    basis = B.basis
    x = approxpoints(basis)
    w = B.weight.(x)
    return approxtransform(basis, w.\f)
end

function iterate_basis(B::Weighted, x)
    Bx = B.basis|x
    b,state = IterTools.@ifsomething iterate(Bx)
    w = B.weight(x)
    return w*b, (w,Bx,state)
end
function iterate_basis(B::Weighted, x, (w,Bx,state))
    b,state = IterTools.@ifsomething iterate(Bx,state)
    return w*b, (w,Bx,state)
end

function iterate_basis(B::Weighted, x::MatrixVectorWrapper)
    xx = MatrixVectorWrapper(x.matrix,B.weight(x))
    Bxx = B.basis|xx
    b,state = IterTools.@ifsomething iterate(Bxx)
    return b, (xx,Bxx,state)
end
function iterate_basis(B::Weighted, x::MatrixVectorWrapper, (xx,Bxx,state))
    b,state = IterTools.@ifsomething iterate(Bxx,state)
    return b, (xx,Bxx,state)
end

evaluate_basis(B::Weighted,i,x) = B.weight(x)*B.basis[i](x)
evaluate_basis(B::Weighted,i,x::MatrixVectorWrapper) = B.basis[i](MatrixVectorWrapper(x.matrix,B.weight(x)))

evaltransform(B::Weighted, x, c) = grideval(B.weight,(x,)) .* evaltransform(B.basis,x,c)
