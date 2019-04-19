"""
    AbstractBasis

Abstract supertype for sets of basis functions.
"""
abstract type AbstractBasis end

"""
    |(B::AbstractBasis, x)

Iterate over the functions in `B` evaluated at point `x`.

# Examples
```
collect( Monomials(4) | 2 ) == [1,2,4,8]
```
"""
Base.:|(B::AbstractBasis,x) = BasisValues(B,wrap(x))

function Base.getindex(B::AbstractBasis, i)
    @assert i in 1:length(B)
    return BasisFunction(B,i)
end

"""
    Matrix(B::AbstractBasis, x) = [ B[j](x[i]) for i = 1:length(x), j = 1:length(B) ]
"""
function Base.Matrix(B::AbstractBasis, x)
    M = Matrix{eltype(collect(B|one(eltype(x))))}(undef, length(x),length(B))
    for i = 1:length(x)
        copyto!(@view(M[i,:]), B|x[i])
    end
    return M
end

"""
    evaltransform(B,x,c)

Evaluate `Matrix(B,x)*c`.
"""
evaltransform(B::AbstractBasis, x, c) = default_evaltransform(B,x,c)
default_evaltransform(B::AbstractBasis, x, c) = Matrix(B,x)*c
default_evaltransform(B::AbstractBasis, x, c::AbstractVector) = mapreduce(((ci,bix),) -> ci*bix, +, zip(c,B|x))
default_evaltransform(B::AbstractBasis, x::AbstractVector{<:Number}, c::AbstractVector) = evaltransform.((B,),x,(c,))


"""
    BasisValues(B,x)

Auxiliary type returned by `B | x`.
"""
struct BasisValues{B<:AbstractBasis,X}
    basis::B
    point::X
end

Base.length(bx::BasisValues) = length(bx.basis)
Base.eltype(bx::BasisValues) = typeof(first(bx))
Base.iterate(bx::BasisValues, args...) = iterate_basis(bx.basis, bx.point, args...)

"""
    iterate_basis(B::AbstractBasis, x [, state])

Iterate over the functions in `B` evaluated at point `x`.
Analogous to `Base.iterate`.
"""
function iterate_basis(B::AbstractBasis, x, i=1)
    i > length(B) && return nothing
    return evaluate_basis(B,i,x), i+1
end


"""
    BasisFunction(B,i)

Auxiliary type returned by `B[i]`.
"""
struct BasisFunction{B<:AbstractBasis}
    basis::B
    i::Int
end
(bf::BasisFunction)(x...) = bf(x)
(bf::BasisFunction)(x) = evaluate_basis(bf.basis, bf.i, wrap(x))
evaluate_basis(B::AbstractBasis, i, x) = nth(B|x,i)



struct LinearCombination{N,C<:AbstractArray{<:Number,N},B<:NTuple{N,AbstractBasis}}
    coeffs::C
    basis::B
    function LinearCombination{N,C,B}(coeffs,basis) where {N,C,B}
        @assert size(coeffs) == length.(basis)
        return new{N,C,B}(coeffs,basis)
    end
end

"""
    LinearCombination(c::AbstractVector, B::AbstractBasis) -> p
    LinearCombination(c::AbstractArray{N}, B::NTuple{N,AbstractBasis}) -> p

Linear combination of basis functions.

# Examples
```
julia> x = LinRange(-1,1,11)
       p = LinearCombination([1,0,2], Monomials(3))
       p.(x) ≈ @.( 1 + 2*x^2 )
true
```
"""
function LinearCombination end
LinearCombination(c::AbstractArray{<:Number,N}, B::NTuple{N,AbstractBasis}) where {N} = LinearCombination{N,typeof(c),typeof(B)}(c,B)
LinearCombination(c::AbstractArray{<:Number,N}, B::Vararg{AbstractBasis,N}) where {N} = LinearCombination(c,B)
LinearCombination(c::AbstractArray{<:Number,N}, B::AbstractBasis) where {N} = LinearCombination(c,ntuple(i->B,Val(N)))

coeffs(c::LinearCombination) = c.coeffs
basis(c::LinearCombination) = c.basis
Base.ndims(c::LinearCombination{N}) where {N} = N
Base.ndims(::Type{<:LinearCombination{N}}) where {N} = N

function evaluate_linear_combination(
    c::AbstractArray{<:Any,N},
    B::NTuple{N,AbstractBasis},
    x::NTuple{N,Any}
) where {N}
    @assert size(c) == length.(B)
    tucker(c, map((B,x)->(c->evaltransform(B,wrap(x),c)), B,x))
end

(p::LinearCombination{N})(x::Vararg{Any,N}) where {N} = evaluate_linear_combination(coeffs(p), basis(p), x)
(p::LinearCombination{N})(x::Vararg{Number,N}) where {N} = first(evaluate_linear_combination(coeffs(p), basis(p), x))

grideval(p::LinearCombination{N}, x::NTuple{N,Any}) where {N} = p(x...)



using MacroTools

"""
    @evaluate expr

Currently supported expressions:
 - `@evaluate p(M)*v`: Apply `p(M)` to `v` efficiently.
"""
macro evaluate(expr)
     @capture(expr, p_(M_)*v_) && return :(evaluate_pmtv($(esc(p)),$(esc(M)),$(esc(v))))
end

evaluate_pmtv(p::LinearCombination,M,v) = p(MatrixVectorWrapper(M,v))
