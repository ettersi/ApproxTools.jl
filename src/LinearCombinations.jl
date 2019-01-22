"""
    Basis

Abstract supertype for sets of basis vectors.
"""
abstract type Basis end
function Base.getindex(b::Basis,i::Integer)
    @assert i in 1:length(b)
    return DefaultBasisFunction(b,i)
end

abstract type BasisValues end
Base.length(bv::BasisValues) = length(bv.basis)
function Base.iterate(bv::BasisValues, i = 1)
    i > length(bv) && return nothing
    return bv[i], i+1
end

abstract type BasisFunction end
using IterTools
struct DefaultBasisFunction{B<:Basis}
    basis::B
    i::Int
end
(bf::DefaultBasisFunction)(x̂) = nth(bf.basis(x̂),bf.i)

"""
    collect(b::Basis,x) -> B

Evaluate `B[i,j] = b[j](x[i])`.
"""
function Base.collect(b::Basis,x::Union{Number,AbstractVector})
    T = promote_type(eltype(b(one(eltype(x)))))
    M = Matrix{T}(undef, length(b),length(x))
    for j = 1:length(x)
        copyto!(@view(M[:,j]), b(x[j]))
    end
    return transpose(M)
end



struct LinearCombination{N,C<:AbstractArray{<:Number,N},B<:NTuple{N,Basis}}
    coefficients::C
    basis::B
end

"""
    LinearCombination(c::AbstractVector, b::Basis) -> p
    LinearCombination(c::AbstractArray{N}, b::NTuple{N,Basis}) -> p

Linear combination of basis functions.
"""
function LinearCombination end
LinearCombination(c::AbstractArray{<:Number,N},b::Basis) where {N}= LinearCombination(c,ntuple(i->b,Val(N)))
GridevalStyle(::Type{<:LinearCombination}) = GridevalCartesian()

coeffs(c::LinearCombination) = c.coefficients
basis(c::LinearCombination) = c.basis
Base.ndims(c::LinearCombination{N}) where {N} = N
Base.ndims(::Type{<:LinearCombination{N}}) where {N} = N

function evaluate_linear_combination(
    c::AbstractVector,
    b::Basis,
    x::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}},
)
    @assert length(c) == length(b)
    bv = b(x)
    T = promote_type(eltype(c),eltype(eltype(bv)))
    return mapreduce(p -> p[1]*p[2], +, zip(c,bv), init=MatFunUtils.zero(T,x))
end

function evaluate_linear_combination(
    c::AbstractArray{<:Any,N},
    b::NTuple{N,Basis},
    x::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    @assert size(c) == length.(b)
    tucker(c, map((b,x)->(f->collect(b,x)*f), b,x))
end

(lc::LinearCombination{N})(x::Vararg{Union{Number,AbstractVector,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}},N}) where {N} = lc(x)
(lc::LinearCombination{N})(x::NTuple{N,Number}) where {N} = evaluate_linear_combination(lc.coefficients, lc.basis, x)[1]
(lc::LinearCombination{N})(x::NTuple{N,Union{Number,AbstractVector}}) where {N} = evaluate_linear_combination(lc.coefficients, lc.basis, x)
(lc::LinearCombination{1})(x::NTuple{1,Number}) = evaluate_linear_combination(lc.coefficients, lc.basis[1], x[1]) # remove ambiguity
(lc::LinearCombination{1})(x::NTuple{1,Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}}) =
    evaluate_linear_combination(lc.coefficients, lc.basis[1], x[1])
(lc::LinearCombination{1})(x::NTuple{1,AbstractVector}) = lc.(x[1])
(lc::LinearCombination{1})(M::AbstractMatrix,v::AbstractVector) = lc((M,v))
