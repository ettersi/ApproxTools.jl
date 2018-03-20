"""
    tucker(C,B)

Multiply each side of tensor `C` with the corresponding matrix `B[k]`.

# Examples
```
julia> C = rand(2); B = (rand(3,2),);
       tucker(C,B) == B[1]*C
true

julia> C = rand(2,3); B = (rand(4,2),rand(5,3));
       tucker(C,B) == B[1]*C*B[2]'
true
```
"""
function tucker(
    C::AbstractArray{<:Any,N},
    B::NTuple{N,AbstractMatrix}
) where {N}
    T = promote_type(eltype(C),eltype.(B)...)
    C = convert(Array{T,N},C)
    # Convert C to its final type here, otherwise type inference gets confused
    for k = 1:N
        tmp = reshape(C,(size(C,1),prod(Base.tail(size(C)))))
        tmp = B[k]*tmp
        C = reshape(convert(Array,transpose(tmp)),(Base.tail(size(C))...,size(B[k],1)))
        # ^ Need to evaluate the transpose, otherwise type inference gets confused
    end
    return C
end

"""
    ×(x...)

Cartesian product.

# Examples
```
julia> [1,2]×[3,4]
2×2 ApproxTools.TensorGrid{...}:
 (1, 3)  (1, 4)
 (2, 3)  (2, 4)
```
"""
×(x::AbstractVector...) = TensorGrid{Tuple{eltype.(x)...},length(x),typeof(x)}(x)
struct TensorGrid{T,N,F} <: AbstractArray{T,N}
    factors::F
end
Base.size(x::TensorGrid) = length.(x.factors)
Base.getindex(x::TensorGrid{<:Any,N,<:Any}, i::Vararg{Int,N}) where {N} = map(getindex,x.factors,i)

"""
    ⊗(x...)

Tensor product.

# Examples
```
julia> [1,2]⊗[3,4]
2×2 ApproxTools.TensorProduct{...}:
 3  4
 6  8
```
"""
function ⊗(x::AbstractVector...)
    grid = ×(x...)
    return TensorProduct{promote_type(eltype.(x)...),length(x), typeof(grid)}(grid)
end
struct TensorProduct{T,N,G} <: AbstractArray{T,N}
    grid::G
end
Base.size(x::TensorProduct) = size(x.grid)
Base.getindex(x::TensorProduct{<:Any,N,<:Any}, i::Vararg{Int,N}) where {N} = prod(x.grid[i...])
