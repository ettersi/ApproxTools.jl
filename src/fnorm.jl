Base.@pure @generated function fndims(f)
    if method_exists(ndims, Tuple{f})
        N = ndims(f)
        return :($N)
    else
        # Function is not type-stable if fndims_via_method_exists
        # is copied here
        return :(fndims_via_method_exists(f))
    end
end
Base.@pure function fndims_via_method_exists(f)
    Base.Cartesian.@nexprs 5 k->begin
        method_exists(f, NTuple{k,Union{}}) && return k
    end
    return 0
end
Base.@pure puremax(a,b) = a > b ? a : b
Base.@pure fndims(f,g) = puremax(fndims(f), fndims(g))

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
    x̂ = (n->2*(0:n-1)/T(n-1)-1).(ntest)
    return vecnorm(grideval(f,x̂),Inf)
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
    x̂ = (n->2*(0:n-1)/T(n-1)-1).(ntest)
    return vecnorm(grideval(f,x̂) .- grideval(g,x̂),Inf)
end
