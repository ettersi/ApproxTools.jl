"""
    grideval(f, x::AbstractVector...)
    grideval(f, x::NTuple{N,AbstractVector})

Evaluate `f` on the grid spanned by `x`.

# Examples
```
julia> grideval(*, ([1,2],[3,4]))
2×2 Array{Int64,2}:
 3  4
 6  8
```
"""
grideval(f, x...) = grideval(f,x)
@generated grideval(f, x::NTuple{N,Any}) where {N} =
    :(Base.Cartesian.@ncall($N,broadcast,f,i->reshape4grideval(x[i],x,i)))

reshape4grideval(xi::Number,x,i) = xi
reshape4grideval(xi::AbstractVector,x,i) = reshape(xi,gridevalshape(x,i))

gridevalshape(x::NTuple{N,Any},i) where {N} = ntuple(j -> i == j ? length(x[i]) : 1, Val(N))
