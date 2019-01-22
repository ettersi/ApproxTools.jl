struct Monomial <: Basis
    n::Int
end

Base.length(b::Monomial) = b.n

evaluationpoints(b::Monomial) = exp.(2π*im/length(b).*(0:length(b)-1))
approxtransform(b::Monomial) = f->fft(convert(Matrix,f),1)./length(b)
# Regarding convert(Matrix,f), see https://github.com/JuliaMath/FFTW.jl/issues/85

(b::Monomial)(x̂::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}) = MonomialValues(b,x̂)
(b::Monomial)(M::AbstractMatrix,v::AbstractVector) = b((M,v))

struct MonomialValues{X̂} <: BasisValues
    basis::Monomial
    evaluationpoint::X̂
end
Base.eltype(::Type{MonomialValues{X̂}}) where {X̂<:Number} = typeof(zero(X̂)*zero(X̂))
Base.eltype(::Type{MonomialValues{X̂}}) where {X̂<:Union{AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}} = MatFunUtils.basis_eltype(X̂)

function Base.iterate(bv::MonomialValues, state=(1,MatFunUtils.dummy(bv.evaluationpoint)))
    x̂ = bv.evaluationpoint
    i,p = state
    i > length(bv) && return nothing
    if i == 1
        p = MatFunUtils.one(x̂)
        return p,(i+1,p)
    else
        p = MatFunUtils.xval(x̂)*p
        return p,(i+1,p)
    end
end

function Base.getindex(b::Monomial,i::Integer)
    @assert i in 1:length(b)
    return MonomialFunction(i)
end

struct MonomialFunction <: BasisFunction
    i::Int
end

(bf::MonomialFunction)(x̂::Union{Number,AbstractMatrix}) = x̂^(bf.i-1)
(bf::MonomialFunction)(x̂::Tuple{AbstractMatrix,AbstractVector}) = DefaultBasisFunction(Monomial(bf.i),bf.i)(x̂)
(bf::MonomialFunction)(M::AbstractMatrix,v::AbstractVector) = bf((M,v))
