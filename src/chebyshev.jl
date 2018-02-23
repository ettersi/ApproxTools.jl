"""
    chebpoints([T,] n)

Compute the `n` Chebyshev points on [-1,1].
"""
chebpoints(T::Type, n::Int) = Chebpoints{T}(n)
chebpoints(n::Int) = chebpoints(Float64,n)
struct Chebpoints{T} <: AbstractVector{T}
    n::Int
    m::T
    b::T
    function Chebpoints{T}(n) where {T}
        if n == 1
            return new(n, T(0), T(π)/2)
        else
            return new(n, -T(π)/(n-1), T(π)*n/(n-1))
        end
    end
end
Base.size(x::Chebpoints) = (x.n,)
Base.getindex(x::Chebpoints, i::Int) = cos(x.m*i+x.b)

baryweights(x::Chebpoints) = ChebBaryWeights{eltype(x)}(length(x))
struct ChebBaryWeights{T} <: AbstractVector{T}
    n::Int
end
Base.size(w::ChebBaryWeights) = (w.n,)
function Base.getindex(w::ChebBaryWeights, i::Int)
    T = eltype(w)
    r = isodd(i) ? one(T) : -one(T)
    if i == 1 || i == w.n
        r /= 2
    end
    return r
end


using FFTW

"""
    chebcoeffs(f)

Evaluate the Chebyshev coefficients the function `f` given in terms
of its values at `n` Chebyshev points.

# Examples

```jldoctest
julia> x = chebpoints(5)

julia> round.( [chebcoeffs(ones(x)) chebcoeffs(x) chebcoeffs(@. 2*x^2-1)], 8)
5×3 Array{Float64,2}:
  1.0   0.0   0.0
 -0.0   1.0   0.0
  0.0  -0.0   1.0
 -0.0  -0.0  -0.0
  0.0  -0.0   0.0
```
"""
function chebcoeffs(f)
    T = eltype(f)
    c = r2r(f,REDFT00)
    f = T(1)/prod(size(c).-1)
    @inbounds for i in CartesianRange(Base.tail(size(c)))
        f̃ = flipsign(f, iseven(count(iseven,i.I)) ? 1 : -1) / 2^count(i->i==1, i.I)
        c[1,i] *= f̃/2
        @simd for i1 = 2:size(c,1)
            c[i1,i] *= flipsign(f̃, isodd(i1) ? 1 : -1)
        end
    end
    return c
end
