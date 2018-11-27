apply(A::AbstractMatrix,B::AbstractMatrix) = A*B
apply(f,B::AbstractMatrix) = f(B)

"""
    tucker(C,B)

Apply `B[k]` to the `k`th dimension of tensor `C`.

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
@generated function tucker(
    C::AbstractArray{<:Any,N},
    B::NTuple{N,Any}
) where {N}
    quote
        C_0 = C
        Base.Cartesian.@nexprs $N k->begin
            tmp_k = reshape(C_{k-1},(size(C_{k-1},1),prod(Base.tail(size(C_{k-1})))))
            tmp_k = apply(B[k],tmp_k)
            C_k = reshape(transpose(tmp_k),(Base.tail(size(C_{k-1}))...,size(tmp_k,1)))
        end
        return $(Symbol("C_",N))
    end
end


abstract type GridevalStyle end
GridevalStyle(f) = GridevalStyle(typeof(f))
GridevalStyle(::Type) = GridevalElementwise()
struct GridevalElementwise <: GridevalStyle end
struct GridevalCartesian <: GridevalStyle end

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
grideval(f, x::Union{Number,AbstractVector}...) = grideval(f,x)
grideval(f, x::NTuple{N,Union{Number,AbstractVector}}) where {N} = grideval(GridevalStyle(f), f, x)
@generated grideval(::GridevalElementwise, f, x::NTuple{N,Union{Number,AbstractVector}}) where {N} =
    :(Base.Cartesian.@ncall($N,broadcast,f,i->reshape4grideval(x[i],x,i)))
grideval(::GridevalCartesian, f, x::NTuple{N,Union{Number,AbstractVector}}) where {N} = f(x)

reshape4grideval(xi::Number,x,i) = xi
reshape4grideval(xi::AbstractVector,x,i) = reshape(xi,gridevalshape(x,i))

gridevalshape(x::NTuple{N,Any},i) where {N} = (i == 1 ? length(x[1]) : 1, gridevalshape(Base.tail(x),i-1)...)
gridevalshape(::NTuple{0},i) = ()
