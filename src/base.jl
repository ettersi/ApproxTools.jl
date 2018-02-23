abstract type InterpolationAlgorithm end

"""
    interpolate(x,f [,y] [,alg]) -> p

Compute the interpolating polynomial to data points `(x,f)`, or
the interpolating rational function with poles `y`. 

This function supports both one dimensional interpolation as well
as interpolation on tensor-product grids in arbitrary dimensions, with
slightly different syntax for the two cases:

    - One dimensional interpolation. `x`, `f` and `y` are `AbstractVector`s,
      `p` allows for pointwise evaluation, e.g. `p(0)`.

    - Interpolation in arbitrary dimension. `x` and `y` are tuples of
      `AbstractVector`s specifying the one-dimensional factors of the
      interpolation grid, `f` is an `AbstractArray`. `p` allows for both
      pointwise evaluation (e.g. `p(0,0)` or `p((0,0))`) as well evaluation
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
interpolate(x::NTuple{N,<:AbstractVector}, f::AbstractArray{<:Any,N}) where {N} = interpolate(x,f,Barycentric)
interpolate(x::AbstractVector,f::AbstractVector, args...) = interpolate((x,),f, args...)
interpolate(x::AbstractVector,f::AbstractVector,y::AbstractVector, args...) = interpolate((x,),f, (y,), args...)


"""
    map2refinterval(a,b)

Compute the affine transformation `A` mapping `[a,b]` to `[-1,1]`.

# Examples

```jldoctest
julia> A = map2refinterval(0,1)

julia> A.*[0,1]
2-element Array{Float64,1}:
 -1.0
  1.0

julia> A\0
0.5
```
"""
map2refinterval(a,b) = AffineTransform(2/(b-a),-(b+a)/(b-a))

struct AffineTransform{T}
    forward::NTuple{2,T}
    backward::NTuple{2,T}
    AffineTransform{T}(m,b) where {T} = new((m,b),(1/m,-b/m))
end
AffineTransform(m::T,b::T) where {T} = AffineTransform{T}(m,b)
AffineTransform(m,b) = AffineTransform(promote(m,b)...)
function Base.:*(at::AffineTransform,x::Number)
    m,b = at.forward
    return m*x+b
end
function Base.:\(at::AffineTransform,x::Number)
    m,b = at.backward
    return m*x+b
end
