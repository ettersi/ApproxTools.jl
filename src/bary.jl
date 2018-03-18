"""
    baryweights(x [,y [,y2]])

Compute the barycentric weights for interpolation points `x` and poles `y`
and `±√(y2)*im`.

More precisely, this function computes the scaled weights `w̃ = w/c`
where `c = exp(mean(log(abs(w)))`. This rescaling prevents over- or
underflow but does not change the result of the barycentric interpolation
formula.
"""
function baryweights(x, y=EmptyVector(), y2=EmptyVector())
    n = length(x)
    T = float(promote_type(eltype.((x,y,y2))...))
    s = ones(T,n)
    a = zeros(T,n)
    for i = 1:n
        @inbounds @simd for j = [1:i-1;i+1:n]
            s[i] *= conj(sign(x[i] - x[j]))
            a[i] -= log(abs(x[i] - x[j]))
        end
        @inbounds @simd for j = 1:length(y)
            s[i] *= sign(x[i] - y[j])
            a[i] += log(abs(x[i] - y[j]))
        end
        @inbounds @simd for j = 1:length(y2)
            s[i] *= sign(x[i]^2 - y2[j])
            a[i] += log(abs(x[i]^2 - y2[j]))
        end
    end
    return @. s*exp(a - $mean(a))
end


"""
    bary(x,w,f,xx)

Evaluate the interpolant to `(x,f)` at `xx` using the
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

    Tden = float(promote_type(eltype(x),eltype(w),eltype(xx)))
    Tnum = promote_type(Tden,eltype(f))

    i = findfirst(x,xx)
    i > 0 && return convert(Tnum,f[i])

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
        Tden = float(promote_type(eltype(x[k]),eltype(w[k]),eltype(xx[k])))
        M = Matrix{Tden}(size(f,1),length(xx[k]))
        @inbounds for j = 1:size(M,2)
            i = findfirst(x[k],xx[k][j])
            if i > 0
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


struct Barycentric <: InterpolationAlgorithm end

function interpolate(
    x::NTuple{N,<:AbstractVector},
    f::AbstractArray{<:Any,N},
    y::NTuple{N,<:AbstractVector},
    y2::NTuple{N,<:AbstractVector},
    ::Barycentric
) where {N}
    @assert length.(x) == size(f)
    w = baryweights.(x,y,y2)
    return BarycentricInterpolant(x,w,f)
end

(p::BarycentricInterpolant{1,<:Any,<:Any,<:Any})(xx::Number) = bary(p.points[1],p.weights[1],p.values,xx)
(p::BarycentricInterpolant{1,<:Any,<:Any,<:Any})(xx...) = throw(MethodError(p,x))
# ^ Make sure we don't accidentally call the below method using e.g. p([0,1])
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx...) where {N} = p(xx)
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx::NTuple{N,Number}) where {N} = first(bary(p.points,p.weights,p.values,xx))
(p::BarycentricInterpolant{N,<:Any,<:Any,<:Any})(xx::NTuple{N,AbstractVector}) where {N} = bary(p.points,p.weights,p.values,xx)
