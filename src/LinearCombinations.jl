"""
    Basis

Abstract supertype for sets of basis functions.
"""
abstract type Basis end

"""
    |(B::Basis, x)

Iterate over the functions in `B` evaluated at point `x`.

# Examples
```
collect( Monomials(4) | 2 ) == [1,2,4,8]
```
"""
Base.:|(B::Basis,x) = BasisValues(B,wrap(x))

function Base.getindex(B::Basis,i::Integer)
    @assert i in 1:length(B)
    return BasisFunction(B,i)
end

"""
    Matrix(B::Basis, x) = [ B[j](x[i]) for i = 1:length(x), j = 1:length(B) ]
"""
function Base.Matrix(B::Basis, x::Union{Number,AbstractVector})
    M = Matrix{eltype(collect(B|one(eltype(x))))}(undef, length(x),length(B))
    for i = 1:length(x)
        copyto!(@view(M[i,:]), B|x[i])
    end
    return M
end

struct BasisValues{B<:Basis,X}
    basis::B
    point::X
end

Base.length(bx::BasisValues) = length(bx.basis)
Base.eltype(bx::BasisValues) = typeof(first(bx))

Base.iterate(bx::BasisValues, args...) = iterate_basis(bx.basis, bx.point, args...)

"""
    iterate_basis(B::Basis, x [, state])

Iterate over the functions in `B` evaluated at point `x`.
Analogous to `Base.iterate`.
"""
function iterate_basis(B::Basis,x, i=1)
    i > length(B) && return nothing
    return evaluate_basis(B,i,x), i+1
end

using IterTools
struct BasisFunction{B<:Basis}
    basis::B
    i::Int
end
(bf::BasisFunction)(x...) = bf(x)
(bf::BasisFunction)(x) = evaluate_basis(bf.basis, bf.i, wrap(x))
evaluate_basis(B::Basis, i, x) = nth(B|x,i)



struct LinearCombination{N,C<:AbstractArray{<:Number,N},B<:NTuple{N,Basis}}
    coeffs::C
    basis::B
end

"""
    LinearCombination(c::AbstractVector, B::Basis) -> p
    LinearCombination(c::AbstractArray{N}, B::NTuple{N,Basis}) -> p

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
LinearCombination(c::AbstractArray{<:Number,N},B::Basis) where {N}= LinearCombination(c,ntuple(i->B,Val(N)))
GridevalStyle(::Type{<:LinearCombination}) = GridevalCartesian()

coeffs(c::LinearCombination) = c.coeffs
basis(c::LinearCombination) = c.basis
Base.ndims(c::LinearCombination{N}) where {N} = N
Base.ndims(::Type{<:LinearCombination{N}}) where {N} = N

function evaluate_linear_combination(
    c::AbstractVector,
    B::Basis,
    x
)
    @assert length(c) == length(B)
    return mapreduce(((ci,bi),) -> ci*bi, +, zip(c,B|x))
end

function evaluate_linear_combination(
    c::AbstractArray{<:Any,N},
    B::NTuple{N,Basis},
    x::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    @assert size(c) == length.(B)
    tucker(c, map((B,x)->Matrix(B,x), B,x))
end



# Wrap arguments into tuple
(p::LinearCombination{N})(x::Vararg{Any,N}) where {N} = p(x)

# One-dimensional case
(p::LinearCombination{1})(x::NTuple{1,Any}) = evaluate_linear_combination(coeffs(p), basis(p)[1], x[1])

# Optimisation: in one dimension, it is faster to evaluate the linear combination pointwise
(p::LinearCombination{1})(x::NTuple{1,AbstractVector}) = p.(x[1])

# Multi-dimensional case
(p::LinearCombination{N})(x::NTuple{N,Union{Number,AbstractVector}}) where {N} = evaluate_linear_combination(coeffs(p), basis(p), x)

# Unpack 1x1 array if all arguments are numbers
(p::LinearCombination{N})(x::NTuple{N,Number}) where {N} = first(invoke(p, Tuple{NTuple{N,Union{Number,AbstractVector}}}, x))

# Resolve ambiguity
(p::LinearCombination{1})(x::NTuple{1,Number}) = first(invoke(p, Tuple{NTuple{1,Any}}, x))



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
