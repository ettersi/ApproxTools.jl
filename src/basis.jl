"""
    Basis

Abstract supertype for sets of basis vectors.
"""
abstract type Basis end

abstract type BasisValues end
Base.length(bv::BasisValues) = length(bv.basis)
Base.start(bv::BasisValues) = 1
Base.next(bv::BasisValues, i) = bv[i], i+1
Base.done(bv::BasisValues, i) = i > length(bv)

function Base.collect(b::Basis,x::Union{Number,AbstractVector})
    T = promote_type(eltype(b(one(eltype(x)))))
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
GridevalStyle(::Type{<:LinearCombination}) = GridevalCartesian()

coeffs(c::LinearCombination) = c.coefficients
basis(c::LinearCombination) = c.basis
Base.ndims(c::LinearCombination{N}) where {N} = N
Base.ndims(::Type{<:LinearCombination{N}}) where {N} = N

"""
    module Utils

Utility functions for abstracting away the differences between
scalar and matrix types.
"""
module Utils
    scalartype(::Type{T}) where {T <: Number} = T
    scalartype(::Type{T}) where {T <: AbstractArray} = eltype(T)

    dummy(x::Union{Number,AbstractMatrix}) = x
    dummy(x::Tuple{AbstractMatrix,AbstractVector}) = x[2]

    zero(::Type{T}, x::Number) where {T} = Base.zero(T)
    zero(::Type{T}, x::AbstractMatrix) where {T} = zeros(T,size(x))
    zero(x::Diagonal) = Diagonal(zero.(diag(x)))
    zero(::Type{T}, x::Tuple{AbstractMatrix,AbstractVector}) where {T} = zeros(T,length(x[2]))

    one(x::Number) = Base.one(x)
    one(x::AbstractMatrix) = eye(x)
    one(x::Diagonal) = Diagonal(Base.one.(diag(x)))
    one(x::Tuple{AbstractMatrix,AbstractVector}) = x[2]

    xval(x::Union{Number,AbstractMatrix}) = x
    xval(x::Tuple{AbstractMatrix,AbstractVector}) = x[1]

    function basis_eltype(::Type{M}) where {M <: AbstractMatrix}
        # Make sure matrix type is closed under multiplication
        @assert Base.return_types(*,Tuple{M,M}) == [M]
        return M
    end
    function basis_eltype(::Type{Tuple{M,V}}) where {M<:AbstractMatrix,V<:AbstractVector}
        # Make sure vector type is closed under multiplication with matrix
        @assert Base.return_types(*,Tuple{M,V}) == [V]
        return V
    end
end

function evaluate_linear_combination(
    c::AbstractVector,
    b::Basis,
    x::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}},
)
    @assert length(c) == length(b)
    bv = b(x)
    T = promote_type(eltype(c),Utils.scalartype(eltype(bv)))
    return mapreduce(p -> p[1]*p[2], +, Utils.zero(T,x), zip(c,bv))
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



struct Semiseparated{N, C, F<:NTuple{N,Any}}
    core::C
    factors::F
end

"""
    Semiseparated(c,f)

Representation of the function `c(x1,x2,...) * f[1](x1) * f[2](x2) * ...`.
"""
Semiseparated(c, f::NTuple{N,Any}) where {N} = Semiseparated{N,typeof(c),typeof(f)}(c,f)

Base.ndims(::Semiseparated{N}) where {N} = N
Base.ndims(::Type{<:Semiseparated{N}}) where {N} = N
GridevalStyle(::Type{<:Semiseparated}) = GridevalCartesian()

(s::Semiseparated{N})(x::Vararg{Union{Number,AbstractVector},N}) where {N} = s(x)
@generated (s::Semiseparated{N})(x::NTuple{N,Number}) where {N} =
    :(Base.Cartesian.@ncall($N,*,s.core(x...), i->s.factors[i](x[i])))
@generated (s::Semiseparated{N})(x::NTuple{N,Union{Number,AbstractVector}}) where {N} =
    :(tucker(grideval(s.core,x), Base.Cartesian.@ntuple($N,i->semiseparated_transform(s.factors[i],x[i]))))

semiseparated_transform(f,x::Number) = c->f(x)*c
semiseparated_transform(f,x::AbstractVector) = c->Diagonal(f.(x))*c


struct Monomial <: Basis
    n::Int
end

Base.length(b::Monomial) = b.n

(b::Monomial)(x̂::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}) = MonomialValues(b,x̂)
(b::Monomial)(M::AbstractMatrix,v::AbstractVector) = b((M,v))

struct MonomialValues{X̂} <: BasisValues
    basis::Monomial
    evaluationpoint::X̂
end
Base.eltype(::Type{MonomialValues{X̂}}) where {X̂<:Number} = typeof(zero(X̂)*zero(X̂))
Base.eltype(::Type{MonomialValues{X̂}}) where {X̂<:Union{AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}} = Utils.basis_eltype(X̂)

function Base.start(bv::MonomialValues)
    x̂ = bv.evaluationpoint
    return 1,Utils.dummy(x̂)
end
function Base.next(bv::MonomialValues, state)
    x̂ = bv.evaluationpoint
    i,p = state
    if i == 1
        p = Utils.one(x̂)
        return p,(i+1,p)
    else
        p = Utils.xval(x̂)*p
        return p,(i+1,p)
    end
end
Base.done(bv::MonomialValues, state) = state[1] > length(bv)


struct Chebyshev <: Basis
    n::Int
end

Base.length(b::Chebyshev) = b.n

interpolationpoints(b::Chebyshev) = chebpoints(b.n)

using FFTW
fftwtype(::Type{T}) where {T <: FFTW.fftwNumber} = T
fftwtype(::Type{T}) where {T <: Real} = Float64
fftwtype(::Type{T}) where {T <: Complex} = ComplexF64
interpolationtransform(b::Chebyshev) = f->begin
    n = length(b)
    T = fftwtype(eltype(f))
    n == 0 && return Array{T}(undef, size(f))
    n == 1 && return convert(Array{T},f)
    c = r2r(f,REDFT00,1)
    d = (real(T)(1)/(n-1)).*(i->isodd(i) ? 1 : -1).(1:n)
    d[1] /= 2; d[end] /= 2
    return Diagonal(d)*c
end

(b::Chebyshev)(x̂::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}) = ChebyshevValues(b,x̂)
(b::Chebyshev)(M::AbstractMatrix,v::AbstractVector) = b((M,v))
struct ChebyshevValues{X̂} <: BasisValues
    basis::Chebyshev
    evaluationpoint::X̂
end
Base.eltype(::Type{ChebyshevValues{X̂}}) where {X̂<:Number} = typeof(zero(X̂)*zero(X̂) + zero(X̂))
Base.eltype(::Type{ChebyshevValues{X̂}}) where {X̂<:Union{AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}} = Utils.basis_eltype(X̂)

function Base.start(bv::ChebyshevValues)
    x̂ = bv.evaluationpoint
    val = Utils.dummy(x̂)
    return 1,val,val
end
function Base.next(bv::ChebyshevValues, state)
    x̂ = bv.evaluationpoint
    i,T0,T1 = state
    if i == 1
        T0,T1 = T1,Utils.one(x̂)
        return T1,(i+1,T0,T1)
    elseif i == 2
        T0,T1 = T1,Utils.xval(x̂)*T1
        return T1,(i+1,T0,T1)
    else
        T0,T1 = T1, 2*Utils.xval(x̂)*T1 - T0
        return T1,(i+1,T0,T1)
    end
end
Base.done(bv::ChebyshevValues, state) = state[1] > length(bv)


struct Weighted{B,W} <: Basis
    basis::B
    weight::W
end

Base.length(b::Weighted) = length(b.basis)

interpolationpoints(b::Weighted) = interpolationpoints(b.basis)

interpolationtransform(b::Weighted) = f->begin
    basis = b.basis
    w = b.weight
    x = interpolationpoints(basis)
    return interpolationtransform(basis)(f./w.(x))
end

(b::Weighted)(x̂::Number) = WeightedValues(b,x̂)

struct WeightedValues{B,V,Ŵ} <: BasisValues
    basis::B
    values::V
    evaluationweight::Ŵ
end
WeightedValues(b,x̂) =  WeightedValues(b,b.basis(x̂),b.weight(x̂))
Base.eltype(::Type{WeightedValues{B,V,Ŵ}}) where {B,V,Ŵ} = promote_type(eltype(V),Ŵ)
Base.start(bv::WeightedValues) = start(bv.values)
function Base.next(bv::WeightedValues, state)
    ŵ = bv.evaluationweight
    p,state = next(bv.values,state)
    return ŵ*p,state
end
Base.done(bv::WeightedValues, state) = done(bv.values,state)
