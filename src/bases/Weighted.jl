"""
    Weighted(B::Basis, w) = [ x-> w(x) * B[k](x) for k = 1:length(B)]
"""
struct Weighted{B,W} <: Basis
    basis::B
    weight::W
end

Base.length(B::Weighted) = length(B.basis)

evaluationpoints(B::Weighted) = evaluationpoints(B.basis)
function approxtransform(B::Weighted, f)
    basis = B.basis
    x = evaluationpoints(basis)
    w = B.weight.(x)
    return approxtransform(basis, w.\f)
end

function iterate_basis(B::Weighted, x)
    b,state = IterTools.@ifsomething iterate_basis(B.basis, x)
    w = B.weight(x)
    return w*b, (w,state)
end
function iterate_basis(B::Weighted, x, (w,state))
    b,state = IterTools.@ifsomething iterate_basis(B.basis, x, state)
    return w*b, (w,state)
end

function iterate_basis(B::Weighted, x::MatrixVectorWrapper)
    xx = MatrixVectorWrapper(x.matrix,B.weight(x))
    b,state = IterTools.@ifsomething iterate_basis(B.basis, xx)
    return b, (xx,state)
end
function iterate_basis(B::Weighted, x::MatrixVectorWrapper, (xx,state))
    b,state = IterTools.@ifsomething iterate_basis(B.basis, xx, state)
    return b, (xx,state)
end

evaluate_basis(B::Weighted,i,x) = B.weight(x)*evaluate_basis(B.basis,i,x)
evaluate_basis(B::Weighted,i,x::MatrixVectorWrapper) = evaluate_basis(B.basis,i,MatrixVectorWrapper(x.matrix,B.weight(x)))

evaltransform(B::Weighted, x, c) = grideval(B.weight,(x,)) .* evaltransform(B.basis,x,c)
