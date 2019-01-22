Base.@pure @generated function fndims(f)
    if hasmethod(ndims, Tuple{f})
        N = ndims(f)
        return :($N)
    else
        # Function is not type-stable if fndims_via_hasmethod
        # is copied here
        return :(fndims_via_hasmethod(f))
    end
end
Base.@pure function fndims_via_hasmethod(f)
    Base.Cartesian.@nexprs 10 k->begin
        hasmethod(f, NTuple{k,Union{}}) && return k
    end
    return 0
end
Base.@pure puremax(a,b) = a > b ? a : b
Base.@pure fndims(f,g) = puremax(fndims(f), fndims(g))

"""
    fnorm(f, ntest=101)
    fnorm(T, f, ntest=101)

Lower-bound the supremum norm of `f` on `[-1,1]^d` by the maximum value
of `abs(f)` on the Cartesian grid of `ntest` equally spaced points in
each dimension.

`ntest` can be either an integer or a tuple of integers. If provided,
`T` specifies the type to be used for the grid points (default `Float64`)

# Examples

```
julia> fnorm(exp)
2.718281828459045

julia> fnorm(Float32, exp)
2.7182817f0

julia> fnorm(x->sin(3x))
0.9999417202299663

julia> fnorm(x->sin(3x),10000)
0.9999999903031747

julia> fnorm( (x1,x2) -> 1/(x1+x2+1/3+im) )
0.999977778518491

julia> fnorm( (x1,x2) -> 1/(x1+x2+1/3+im), (1000,1000) )
0.9999994989988744
```
"""
fnorm(args...) = fnorm(Float64, args...)
function fnorm(
    ::Type{T},
    f,
    ntest::Integer = 101,
    ndims::Val{N} = Val(fndims(f))
) where {T,N}
    @assert N > 0
    fnorm(T,f,ntuple(i->ntest,ndims))
end
function fnorm(
    ::Type{T},
    f,
    ntest::NTuple{N,Integer}
) where {T,N}
    x̂ = (n->LinRange{T}(-1,1,n)).(ntest)
    return norm(grideval(f,x̂),Inf)
end

function fnorm(
    ::Type{T},
    f,g,
    ntest::Integer = 101,
    ndims::Val{N} = Val(fndims(f,g))
) where {T,N}
    @assert N > 0
    fnorm(T,f,g,ntuple(i->ntest,ndims))
end
function fnorm(
    ::Type{T},
    f,g,
    ntest::NTuple{N,Integer}
) where {T,N}
    x̂ = (n->LinRange{T}(-1,1,n)).(ntest)
    return norm(grideval(f,x̂) .- grideval(g,x̂),Inf)
end
