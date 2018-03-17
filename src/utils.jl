struct EmptyArray{N} <: AbstractArray{Union{},N} end
Base.size(::EmptyArray{N}) where {N} = ntuple(i->0, Val{N}())
Base.getindex(v::EmptyArray{N},i::Vararg{Int,N}) where {N} = throw(BoundsError(v,i))
const EmptyVector = EmptyArray{1}
const EmptyMatrix = EmptyArray{2}


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
    AffineTransform{T}(m,b) where {T} = new((m,b),(inv(m),-m\b))
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
