using IterTools


"""
    geometric_mean_distance(x)

Compute the geometric mean distance between the points `x`.

# Examples

```jldoctest
julia> x = rand(3);
julia> geometric_mean_distance(x) ≈ abs( (x[1]-x[2])*(x[1]-x[3])*(x[2]-x[3]) )^(1/3)
true
"""
function geometric_mean_distance(x)
    if length(x) <= 1
        return one(typeof(geometric_mean_distance(zeros(eltype(x),2))))
    else
        n = length(x)
        return exp(sum(log(abs(x[i]-x[j])) for (i,j) in subsets(1:n,2))*2/(n*(n-1)))
    end
end

"""
    baryweights(x)

Compute the barycentric weights associated with the interpolation points `x`.

More precisely, this function computes the scaled weights `w̃ = w/d^(n-1)`
where `d = geometric_mean_distance(x)`. This rescaling prevents over- or
underflow but does not change the result of the barycentric interpolation
formula.
"""
function baryweights(x)
    # Following baryWeights.m from Chebfun.
    @assert length(x) > 0
    n = length(x)
    x /= geometric_mean_distance(x)
    w = Vector{eltype(x)}(n)
    for i = 1:n 
        s = one(sign(x[1]-x[1]))
        l = zero(log(abs(x[1]-x[1])))
        @inbounds for j = 1:n
            j == i && continue
            s *=    sign(x[i]-x[j])
            l += log(abs(x[i]-x[j]))
        end
        w[i] = s*exp(-l)
    end
    return w
end

"""
    bary(x,w,f,xx)

Evaluate the polynomial interpolant to `(x,f)` at `xx` using the
barycentric interpolation formula with weights `w`.

Two implementations of `bary(x,w,f,xx)` are provided.
  - One dimensional interpolation evaluated at a single point. In this
    case, `x`, `w` and `f` are `AbstractVector`s and `xx` is a `Number`.
  - Tensor-product interpolation evaluated at a single point or on a
    cartesian grid. In this case, `x` and `w` are tuples of `AbstractVector`s
    specifying the one-dimensional factors of the interpolation grid and the
    corresponding barycentric weights, respectively. `f` is an `AbstractArray`,
    and `xx` is a tuple of `Number`s or `AbstractVector`s containing the one-
    dimensional factors of the evaluation points.

# Examples

```jldoctest
julia> x = [0,1]
       w = baryweights(x)
       f = [0,1]
       bary(x,w,f,0.5)
0.5
```

```jldoctest
julia> x = ([0,1],[0,1])
       w = baryweights.(x)
       f = [0 0; 0 1]
       bary(x,w,f,(0.5,0.5))
1×1 Array{Float64,2}:
 0.25
```

```jldoctest
julia> x = ([0,1],[0,1])
       w = baryweights.(x)
       f = [0 0; 0 1]
       bary(x,w,f,([0,0.5,1],[0,0.5,1]))
3×3 Array{Float64,2}:
 0.0  0.0   0.0
 0.0  0.25  0.5
 0.0  0.5   1.0
```
"""
function bary(
    x::AbstractVector,
    w::AbstractVector,
    f::AbstractVector,
    xx::Number
)
    @assert length(x) == length(w) == length(f)
    n = length(f)

    Tden = typeof(w[1]/(xx-x[1]))
    Tnum = typeof(zero(Tden)*f[1])

    i = findfirst(x,xx)
    if i <= n && xx == x[i]
        return convert(Tnum,f[i])
    end

    num = zero(Tnum)
    den = zero(Tden)
    @inbounds @simd for i = 1:n
        t = w[i]/(xx-x[i])
        num += t*f[i]
        den += t
    end
    return num/den
end

function bary(
    x::NTuple{N,AbstractVector},
    w::NTuple{N,AbstractVector},
    f::AbstractArray{<:Any,N},
    xx::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    @assert length.(x) == length.(w) == size(f)
    for k = 1:N
        Tden = typeof(w[k][1]/(xx[k][1]-x[k][1]))
        M = Matrix{Tden}(size(f,1),length(xx[k]))
        @inbounds for j = 1:size(M,2)
            i = searchsortedfirst(x[k],xx[k][j])
            if i <= length(x[k]) && x[k][i] == xx[k][j]
                M[:,j] .= 0
                M[i,j] = 1
            else
                d = zero(Tden)
                @simd for i = 1:size(M,1)
                    t = w[k][i]/(xx[k][j]-x[k][i])
                    M[i,j] = t
                    d += t
                end
                M[:,j] .*= 1/d
            end
        end

        sf = size(f)
        tmp = reshape(f,(sf[1],prod(Base.tail(sf))))
        tmp = M'*tmp
        f = reshape(tmp',(Base.tail(sf)...,size(M,2)))
    end
    return f
end


struct BarycentricInterpolant{N,X,W,F}
    points::X
    weights::W
    values::F
end
BarycentricInterpolant(x,w,f) = BarycentricInterpolant{ndims(f),typeof(x),typeof(w),typeof(f)}(x,w,f)

Base.ndims(::Type{BarycentricInterpolant{N,<:Any,<:Any,<:Any}}) where {N} = N
Base.ndims(::BarycentricInterpolant{N,<:Any,<:Any,<:Any}) where {N} = N


struct BarycentricInterpolationAlgorithm <: InterpolationAlgorithm end
const Barycentric = BarycentricInterpolationAlgorithm()

function interpolate(
    x::NTuple{N,<:AbstractVector},
    f::AbstractArray{<:Any,N},
    ::BarycentricInterpolationAlgorithm
 ) where {N}
    @assert length.(x) == size(f)
    @assert all(size(f) .> 0)
    w = baryweights.(x)
    BarycentricInterpolant(x, w, f)
end

(p::BarycentricInterpolant{1,<:Any,<:Any,<:Any})(xx::Number) = bary(p.points[1],p.weights[1],p.values,xx)
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx...) where {N} = p(xx)
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx::NTuple{N,Number}) where {N} = first(bary(p.points,p.weights,p.values,xx))
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx::NTuple{N,AbstractVector}) where {N} = bary(p.points,p.weights,p.values,xx)
