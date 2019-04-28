Base.@pure @generated function fndims(f)
    if hasmethod(ndims, Tuple{f})
        N = ndims(f)
        return :($N)
    else
        # Function is not type-stable if fndims_via_hasmethod
        # is copied here
        return quote
            (d = fndims_via_hasmethod(f, Union{})) > 0 && return d
            (d = fndims_via_hasmethod(f, Float64)) > 0 && return d
            (d = fndims_via_hasmethod(f, ComplexF64)) > 0 && return d
            (d = fndims_via_hasmethod(f, Float32)) > 0 && return d
            (d = fndims_via_hasmethod(f, ComplexF32)) > 0 && return d
            (d = fndims_via_hasmethod(f, Int)) > 0 && return d
            throw(ArgumentError("Could not determine the dimension of function $f"))
        end
    end
end
Base.@pure function fndims_via_hasmethod(f,T)
    Base.Cartesian.@nexprs 10 k->begin
        hasmethod(f, NTuple{k,T}) && return k
    end
    return 0
end


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
grideval(f, x) = grideval(f,ntuple(i->x,Val(fndims(f))))
grideval(f, x...) = grideval(f,x)
grideval(f, x::NTuple{N,Any}) where {N} = f.(reshape4grideval(x)...)

reshape4grideval(x::NTuple{N,Any}) where {N} = reshape4grideval_.(x, Val(N), ntuple(identity, Val(N)))
reshape4grideval_(xi::Number,_,_) = xi
reshape4grideval_(xi::AbstractVector,::Val{N},i) where {N} = reshape(xi,ntuple(j -> i==j ? length(xi) : 1, Val(N)))



using MacroTools

struct GridFunction{N,F}
    fun::F
end
Base.ndims(::GridFunction{N}) where {N} = N
Base.ndims(::Type{<:GridFunction{N}}) where {N} = N
GridFunction{N}(f) where {N} = GridFunction{N,typeof(f)}(f)

(gf::GridFunction{N})(x::Vararg{Any,N}) where {N} = gf.fun(x...)
grideval(gf::GridFunction{N}, x::NTuple{N,Any}) where {N} = gf.fun(reshape4grideval(x)...)

cvec(x::Number) = x
cvec(x::AbstractArray) = vec(x)

"""
    @gridfun(f)

Assemble an anonymous function where nested function calls
annotated with `\$` get `grideval`ed separately.

# Examples

`grideval(@gridfun((x1,x2)->f(x1,x2) - \$g(x1,x2)),x)` is equivalent to

    grideval((x1,x2)->f(x1,x2) - grideval(g,x),x)
"""
macro gridfun(f)
    @capture(f, (x_) -> body_) || error("@gridfun must be used on an anonymous function")
    body = MacroTools.postwalk(body) do expr
        (@capture(expr, g_(y__)) && isa(g,Expr) && g.head == :$) || return expr
        return :($grideval($g, $cvec.(($(y...),))))
    end
    if isa(x, Expr) N = length(x.args)
    else N = 1; end
    return esc(:($GridFunction{$N}($x -> @. $body)))
end
