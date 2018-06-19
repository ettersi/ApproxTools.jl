"""
    tucker(C,B)

Apply `B[k]` to the `k`th dimension of tensor `C`.

# Examples
```
julia> C = rand(2); B = (rand(3,2),);
       tucker(C,(B->(C->B*C)).(B)) == B[1]*C
true

julia> C = rand(2,3); B = (rand(4,2),rand(5,3));
       tucker(C,(B->(C->B*C)).(B)) == B[1]*C*B[2]'
true
```
"""
@generated function tucker(
    C::AbstractArray{<:Any,N},
    B::NTuple{N,Any}
) where {N}
    quote
        C_0 = C
        Base.Cartesian.@nexprs $N k->begin
            tmp_k = reshape(C_{k-1},(size(C_{k-1},1),prod(Base.tail(size(C_{k-1})))))
            tmp_k = B[k](tmp_k)
            C_k = reshape(transpose(tmp_k),(Base.tail(size(C_{k-1}))...,size(tmp_k,1)))
        end
        return $(Symbol("C_",N))
    end
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
@generated grideval(f, x::NTuple{N,AbstractVector}) where {N} = :(Base.Cartesian.@ncall($N,broadcast,f,i->reshape(x[i],shape(x,i))))
grideval(f::LinearCombination{N}, x::NTuple{N,AbstractVector}) where {N} = f(x)

# Utility function for grideval
# x is only used for its length. It would be cleaner to pass Val(N) instead, but unlike
# Base.tail, Val(N-1) is not type-stable.
shape(x::NTuple{N,Any},i) where {N} = (shape(Base.tail(x),i)..., i == N ? length(x[i]) : 1)
shape(::NTuple{0},i) = ()
