abstract type InnerProduct end

dot_arg(f,p::InnerProduct) = evaluate(f,p)
dot_arg(f::AbstractArray,p::InnerProduct) = [f(x) for x in evalpoints(p), f in f]
dot_arg(f::AbstractBasis,p::InnerProduct) = Matrix(f,evalpoints(p))

LinearAlgebra.norm(f, p::InnerProduct) = norm(dot_arg(f,p), p)
LinearAlgebra.dot(f,g,p::InnerProduct) = dot(dot_arg.((f,g),Ref(p))...,p)

"""
    WeightedL2(x,w) <: InnerProduct

# Example
```
p = WeightedL2([0,1],[2,3])
norm(identity,p) -> sqrt(0^2*2 + 1^2*3)
```
"""
struct WeightedL2{X,W} <: InnerProduct
    points::X
    weights::W
end

evalpoints(p::WeightedL2) = p.points
evaluate(f,p::WeightedL2) = f.(evalpoints(p))
dot_arg(f::AbstractArray{<:Number},p::WeightedL2) = f

LinearAlgebra.norm(f::AbstractVecOrMat{<:Number}, p::WeightedL2) = sqrt(dot(f,f,p))

function LinearAlgebra.dot(
    f::AbstractVecOrMat{<:Number},
    g::AbstractVecOrMat{<:Number},
    p::WeightedL2
)
    w = p.weights
    # Workaround for https://github.com/JuliaLang/julia/issues/35424
    length(w) == 0 && return zero(eltype(f))*zero(eltype(w))*zero(eltype(g))
    return f'*Diagonal(w)*g
end


using FastGaussQuadrature

L2(n) = WeightedL2(gausslegendre(n)...)
