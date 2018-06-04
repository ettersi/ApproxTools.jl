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
    B::NTuple{N,Any}
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
    cartesian(x::AbstractVector...)
    cartesian(x::NTuple{N,AbstractVector})

Cartesian product.

# Examples
```
julia> cartesian([1,2],[3,4])
2×2 ApproxTools.TensorGrid{...}:
 (1, 3)  (1, 4)
 (2, 3)  (2, 4)
```
"""
cartesian(x::AbstractVector...) = cartesian(x)
cartesian(x::NTuple{N,AbstractVector}) where {N} = TensorGrid{Tuple{eltype.(x)...},length(x),typeof(x)}(x)
struct TensorGrid{T,N,F} <: AbstractArray{T,N}
    factors::F
end
Base.size(x::TensorGrid) = length.(x.factors)
Base.getindex(x::TensorGrid{<:Any,N,<:Any}, i::Vararg{Int,N}) where {N} = map(getindex,x.factors,i)

"""
    grideval(f, x::AbstractVector...)
    grideval(f, x::NTuple{N,AbstractVector})

Evaluate `f` on the grid spanned by the `x`.

# Examples
```
julia> grideval(*, ([1,2],[3,4]))
2×2 Array{Int64,2}:
 3  4
 6  8
```
"""
grideval(f, x::AbstractVector...) = grideval(f,x)
grideval(f, x::NTuple{N,AbstractVector}) where {N} = (x->f(x...)).(cartesian(x))
