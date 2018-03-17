abstract type InterpolationAlgorithm end

"""
    interpolate(x,f [,y [,y2]] [,alg]) -> p

Compute the interpolating polynomial to data points `(x,f)`, or
the interpolating rational function with poles `y` and `±√(y2)*im`.

This function supports both one dimensional interpolation as well
as interpolation on tensor-product grids in arbitrary dimensions, with
slightly different syntax for the two cases:

    - One dimensional interpolation. `x`, `f`, `y` and `y2` are
      `AbstractVector`s, `p` allows for pointwise evaluation, e.g. `p(0)`.

    - Interpolation in arbitrary dimension. `x`, `y` and `y2` are tuples of
      `AbstractVector`s specifying the one-dimensional factors of the
      interpolation grid, `f` is an `AbstractArray`. `p` allows for both
      pointwise evaluation (e.g. `p(0,0)`) as well evaluation
      on tensor-product grids (e.g. `p(([0,1],[0,1]))`).

# Examples

```jldoctest
julia> x = [0,1]
       f = [0,1]
       p = interpolate(x,f)
       p(0.5)
0.5
```

```jldoctest
julia> x = ([0,1],[0,1])
       f = [0 0; 0 1]
       p = interpolate(x,f)
       p(0.5,0.5)
1×1 Array{Float64,2}:
 0.25
```

```jldoctest
julia> x = ([0,1],[0,1])
       f = [0 0; 0 1]
       p = interpolate(x,f)
       p(([0,0.5,1],[0,0.5,1]))
3×3 Array{Float64,2}:
 0.0  0.0   0.0
 0.0  0.25  0.5
 0.0  0.5   1.0
```
"""
function interpolate end

# Fill in the poles
interpolate(
    x::NTuple{N,<:AbstractVector},
    f::AbstractArray{<:Any,N},
    alg::InterpolationAlgorithm = Barycentric()
) where {N} = interpolate(
    x,f,
    ntuple(i->EmptyVector(),Val{N}()),
    ntuple(i->EmptyVector(),Val{N}()),
    alg
)

interpolate(
    x::NTuple{N,<:AbstractVector},
    f::AbstractArray{<:Any,N},
    y::NTuple{N,<:AbstractVector},
    alg::InterpolationAlgorithm = Barycentric()
) where {N} = interpolate(
    x,f,y,
    ntuple(i->EmptyVector(),Val{N}()),
    alg
)

interpolate(
    x::NTuple{N,<:AbstractVector},
    f::AbstractArray{<:Any,N},
    y::NTuple{N,<:AbstractVector},
    y2::NTuple{N,<:AbstractVector}
) where {N} = interpolate(x,f,y,y2,Barycentric())

# Map from 1D to ND
# The args... is only meant to capture the alg parameter
interpolate(
    x::AbstractVector,
    f::AbstractVector,
    args...
) = interpolate((x,),f, args...)
interpolate(
    x::AbstractVector,
    f::AbstractVector,
    y::AbstractVector,
    args...
) = interpolate((x,),f, (y,), args...)
interpolate(
    x::AbstractVector,
    f::AbstractVector,
    y::AbstractVector,
    y2::AbstractVector,
    args...
) = interpolate((x,),f, (y,), (y2,), args...)
