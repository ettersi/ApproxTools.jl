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
grideval(f, x::NTuple{N,Any}) where {N} = f.(reshape4grideval(x)...)

reshape4grideval(x::NTuple{N,Any}) where {N} = reshape4grideval_.(x, Val(N), ntuple(identity, Val(N)))
reshape4grideval_(xi::Number,_,_) = xi
reshape4grideval_(xi::AbstractVector,::Val{N},i) where {N} = reshape(xi,ntuple(j -> i==j ? length(xi) : 1, Val(N)))



using MacroTools

struct GridFunction{N,F}
    fun::F
end
GridFunction{N}(f) where {N} = GridFunction{N,typeof(f)}(f)

(gf::GridFunction{N})(x::Vararg{Any,N}) where {N} = gf.fun(x...)
grideval(gf::GridFunction{N}, x::NTuple{N,Any}) where {N} = gf.fun(reshape4grideval(x)...)

cvec(x::Number) = x
cvec(x::AbstractArray) = vec(x)

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
