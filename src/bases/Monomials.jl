struct Monomials <: Basis
    n::Int
end

Base.length(b::Monomials) = b.n

evaluationpoints(b::Monomials) = exp.(2π*im/length(b).*(0:length(b)-1))
approxtransform(b::Monomials) = f->fft(convert(Matrix,f),1)./length(b)
# Regarding convert(Matrix,f), see https://github.com/JuliaMath/FFTW.jl/issues/85

(b::Monomials)(x̂::Union{Number,AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}) = MonomialsValues(b,x̂)
(b::Monomials)(M::AbstractMatrix,v::AbstractVector) = b((M,v))

struct MonomialsValues{X̂} <: BasisValues
    basis::Monomials
    evaluationpoint::X̂
end
Base.eltype(::Type{MonomialsValues{X̂}}) where {X̂<:Number} = typeof(zero(X̂)*zero(X̂))
Base.eltype(::Type{MonomialsValues{X̂}}) where {X̂<:Union{AbstractMatrix,Tuple{AbstractMatrix,AbstractVector}}} = MatFun.basis_eltype(X̂)

function Base.iterate(bv::MonomialsValues)
    x̂ = bv.evaluationpoint
    p = MatFun.one(x̂)
    return p,(2,p)
end
function Base.iterate(bv::MonomialsValues, state)
    x̂ = bv.evaluationpoint
    i,p = state
    i > length(bv) && return nothing
    p = MatFun.xmul(x̂)*p
    return p,(i+1,p)
end

function Base.getindex(b::Monomials,i::Integer)
    @assert i in 1:length(b)
    return MonomialsFunction(i)
end

struct MonomialsFunction <: BasisFunction
    i::Int
end

(bf::MonomialsFunction)(x̂::Union{Number,AbstractMatrix}) = x̂^(bf.i-1)
(bf::MonomialsFunction)(x̂::Tuple{AbstractMatrix,AbstractVector}) = DefaultBasisFunction(Monomials(bf.i),bf.i)(x̂)
(bf::MonomialsFunction)(M::AbstractMatrix,v::AbstractVector) = bf((M,v))
