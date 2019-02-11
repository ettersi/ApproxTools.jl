"""
    Weighted(B::Basis, w) = [ x-> w(x) * B[k](x) for k = 1:length(B)]
"""
struct Weighted{B,W} <: Basis
    basis::B
    weight::W
end

Base.length(B::Weighted) = length(B.basis)

evaluationpoints(B::Weighted) = evaluationpoints(B.basis)
approxtransform(B::Weighted) = f->begin
    basis = B.basis
    x = evaluationpoints(basis)
    w = B.weight.(x)
    return apply(approxtransform(basis), w.\f)
end

function iterate_basis(B::Weighted, x)
    tmp = iterate_basis(B.basis, x)
    tmp == nothing && return nothing
    p,state = tmp
    w = B.weight(x)
    return w*p, (w,state)
end
function iterate_basis(B::Weighted, x, (w,state))
    tmp = iterate_basis(B.basis, x, state)
    tmp == nothing && return nothing
    p,state = tmp
    return w*p, (w,state)
end

iterate_basis(B::Weighted, x::MatrixVectorWrapper) = iterate_basis(B.basis, MatrixVectorWrapper(x.matrix,B.weight(x)))
iterate_basis(B::Weighted, x::MatrixVectorWrapper, state) = iterate_basis(B.basis, x, state)

evaluate_basis(B::Weighted,i::Integer,x) = B.weight(x)*evaluate_basis(B.basis,i,x)
evaluate_basis(B::Weighted,i::Integer,x::MatrixVectorWrapper) = evaluate_basis(B.basis,i,MatrixVectorWrapper(x.matrix,B.weight(x)))

evaltransform(B::Weighted, x::Union{Number, AbstractVector}) = c -> grideval(B.weight,(x,)) .* apply(evaltransform(B.basis,x), c)
# TODO: What about evaluate_linear_combination(c,b::NTuple{Weighted},x::NTuple{1,AbstractVector}) ? 
