abstract type InnerProduct end

LinearAlgebra.norm(f, p::InnerProduct) = norm(dot_arg(f,evalpoints(p)), p)
LinearAlgebra.norm(fx::AbstractVector{<:Number}, p::InnerProduct) = sqrt(dot(fx,fx,p))

dot_arg(f,x) = f.(x)
dot_arg(f::AbstractArray{<:Number},x) = f
dot_arg(f::AbstractArray,x) = [f(x) for x in x, f in f]
dot_arg(f::AbstractBasis,x) = Matrix(f,x)

function LinearAlgebra.dot(f,g,p::InnerProduct)
    fx,gx = dot_arg.((f,g),Ref(evalpoints(p)))
    return dot(fx,gx,p)
end

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

function LinearAlgebra.dot(
    f::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    g::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}},
    p::WeightedL2
)
    w = p.weights
    return f'*Diagonal(w)*g
end


using FastGaussQuadrature

L2(n) = WeightedL2(gausslegendre(n)...)
