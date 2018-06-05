"""
    Basis

Abstract supertype for sets of basis vectors.
"""
abstract type Basis end
Base.eltype(b::Basis,x̂::Number) = eltype(typeof(b),typeof(x̂))

abstract type BasisValues{Basis, Evaluationpoint} end
Base.length(bv::BasisValues) = length(bv.basis)
Base.eltype(::Type{BasisValues{B,X̂}}) where {B,X̂} = eltype(B,X̂)
Base.start(bv::BasisValues) = 1
Base.next(bv::BasisValues, i) = bv[i], i+1
Base.done(bv::BasisValues, i) = i > length(bv)

function Base.collect(b::Basis,x::Union{Number,AbstractVector})
    T = promote_type(eltype(typeof(b),eltype(x)))
    M = Matrix{T}(undef, length(b),length(x))
    for j = 1:length(x)
        copyto!(@view(M[:,j]), b(x[j]))
    end
    return transpose(M)
end


"""
    interpolationpoints(b::Basis)

Interpolation points. See [`interpolate()`](@ref).
"""
function interpolationpoints end

"""
    interpolationtransform(b::Basis, x::AbstractVector)

Interpolation transform. See [`interpolate()`](@ref).
"""
function interpolationtransform end

"""
    interpolate(f, b::Basis) -> LinearCombination{1}
    interpolate(f, b::NTuple{N,Basis}) -> LinearCombination{N}
"""
interpolate(f, b::Basis) = interpolate(f,(b,))
interpolate(f, b::NTuple{N,Basis}) where {N} =
    LinearCombination(
        tucker(
            grideval(f,interpolationpoints.(b)),
            interpolationtransform.(b)
        ),
        b
    )



struct LinearCombination{N,C<:AbstractArray{<:Number,N},B<:NTuple{N,Basis}}
    coefficients::C
    basis::B
end

"""
    LinearCombination(c::AbstractVector, b::Basis) -> p
    LinearCombination(c::AbstractArray{N}, b::NTuple{N,Basis}) -> p

Linear combination of basis functions, `p(x) = sum(c.*b(x))`.

The returned function object allows for pointwise evaluation (e.g.
`p(x)` or `p(x,y)`) or evaluation on tensor-product grids for
dimensions `N > 1` (e.g. `p([x1,x2],[y1,y2]) -> Matrix`).
"""
function LinearCombination end
LinearCombination(c::AbstractArray{<:Number,N},b::NTuple{N,Basis}) where {N} =
    LinearCombination{N,typeof(c),typeof(b)}(c,b)
LinearCombination(c::AbstractArray{<:Number,1},b::Basis) = LinearCombination(c,(b,))

coeffs(c::LinearCombination) = c.coefficients
basis(c::LinearCombination) = c.basis

function evaluate_linear_combination(
    c::AbstractVector,
    b::Basis,
    x::Number
)
    @assert length(c) == length(b)
    T = promote_type(eltype(c),eltype(b,x))
    return mapreduce(p -> p[1]*p[2], +, zero(T), zip(c,b(x)))
end

function evaluate_linear_combination(
    c::AbstractArray{<:Any,N},
    b::NTuple{N,Basis},
    x::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    @assert size(c) == length.(b)
    tucker(c, map((b,x)->(f->collect(b,x)*f), b,x))
end

(lc::LinearCombination{N})(x::Vararg{Union{Number,AbstractVector},N}) where {N} = lc(x)
(lc::LinearCombination{N})(x::NTuple{N,Number}) where {N} = evaluate_linear_combination(lc.coefficients, lc.basis, x)[1]
(lc::LinearCombination{N})(x::NTuple{N,Union{Number,AbstractVector}}) where {N} = evaluate_linear_combination(lc.coefficients, lc.basis, x)
(lc::LinearCombination{1})(x::NTuple{1,Number}) = evaluate_linear_combination(lc.coefficients, lc.basis[1], x[1])
(lc::LinearCombination{1})(x::NTuple{1,AbstractVector}) = throw(MethodError(lc,x))
