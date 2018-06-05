"""
    LogNumber{S,L}

Represent a real or complex number in terms of its sign and logarithm.

This is a dirty hack around the limitation that standard floating-point
types link the bitcount of the exponent to that of the mantissa, which
makes it impossible to represent very large or small values without
simultaneously (and unnecessarily) increasing the precision of the
representation. The drawback of this hack is that values x close to 1
have a larger relative accuracy than values far away from 1 since

    1 - exp(l*(1±ε)) / exp(l) == 1 - exp(±l*ε) ≈ ∓l*ε.

It remains to be seen whether this has any practical relevance.

A better workaround would be to create a floating-point with
arbitrary exponent and mantissa.
"""
struct LogNumber{S<:Union{AbstractFloat,Complex{<:AbstractFloat}},L<:AbstractFloat} <: Number
    sign::S
    logabs::L
end
LogNumber(s::Number, l::Real) = LogNumber{float(typeof(s)),float(typeof(l))}(s,l)

Base.sign(x::LogNumber) = x.sign
logabs(x::Number) = log(abs2(x))/2
logabs(x::LogNumber) = x.logabs

lognumber(::Type{T}) where {T<:Number} = LogNumber{float(T), float(real(T))}
lognumber(x::Number) = convert(LogNumber, x)
Base.float(x::LogNumber{S,L}) where {S,L} = convert(promote_type(S,L),x)

Base.one(::Type{LogNumber{S,L}}) where {S,L} = LogNumber{S,L}(1,0)

Base.convert(::Type{LogNumber}, x::LogNumber) = x
Base.convert(::Type{LogNumber{S,L}}, x::LogNumber) where {S,L} = LogNumber{S,L}(x.sign,x.logabs)
Base.convert(::Type{LogNumber}, x::Number) = LogNumber(float(sign(x)),logabs(x))
Base.convert(::Type{LogNumber{S,L}}, x::Number) where {S,L} = LogNumber{S,L}(sign(x),logabs(x))
Base.convert(::Type{T}, x::LogNumber) where {T<:Union{AbstractFloat,Complex{<:AbstractFloat}}} = convert(T,x.sign)*exp(convert(real(T),x.logabs))

Base.promote_rule(::Type{LogNumber{S,L}}, ::Type{T}) where {S,L,T<:Number}=
    LogNumber{promote_type(S,T), promote_type(L,float(real(T)))}
Base.promote_rule(::Type{LogNumber{S1,L1}}, ::Type{LogNumber{S2,L2}}) where {S1,L1,S2,L2}=
    LogNumber{promote_type(S1,S2), promote_type(L1,L2)}

Base.:*(x::LogNumber,y::LogNumber) = LogNumber(x.sign * y.sign, x.logabs + y.logabs)
Base.:/(x::LogNumber,y::LogNumber) = LogNumber(x.sign * conj(y.sign), x.logabs - y.logabs)


"""
    prodpot(x̂::Number,x::AbstractVector)

Efficient and stable implementation of `prod(x̂-x)`.

The result is returned as a `LogNumber` to avoid over- or underflow.
"""
function prodpot(x̂::Number,x::AbstractVector)
    T = lognumber(float(promote_type(typeof(x̂),eltype(x))))
    return mapreduce(xj->x̂-xj, *, one(T), x)
end
"""
    prodpot(x::AbstractVector)

Efficient and stable implementation of

    [ prod(x[i] - x[setdiff(1:length(x),i)]) for i = 1:length(x) ]

The result is returned as a `Vector{LogNumber}` to avoid over- or
underflow.
"""
function prodpot(x::AbstractVector)
    n = length(x)
    return (i -> prodpot(x[i], @view x[[1:i-1; i+1:n]])).(1:n)
end


struct Barycentric{X,W,Λ} <: Basis
    interpolationpoints::X
    weight::W
    baryweights::Λ
end

"""
    Barycentric(x, pot = one)

Basis functions for barycentric interpolation.

The basis functions are given by

   b[k](x̂) = prod(x̂ - x̃) / prod(x[k] - x̃) * pot(x̂) / pot(x[k])

where

    x̃ = x[setdiff(1:n,k)]
"""
Barycentric(x::AbstractVector, w = one) = Barycentric(x,w,1./(w.(x).*prodpot(x)))

Base.length(b::Barycentric) = length(b.interpolationpoints)
Base.eltype(::Type{Barycentric{X,W,Λ}},::Type{X̂}) where {X,W,Λ,X̂} = promote_type(float(eltype(Λ)),X̂)

interpolationpoints(b::Barycentric) = b.interpolationpoints
interpolationtransform(b::Barycentric) = identity

function (b::Barycentric)(x̂::Number)
    x = b.interpolationpoints
    w = b.weight
    l = w(x̂) * prodpot(x̂,x)
    idx = findfirst(x,x̂)
    return BarycentricValues{eltype(b,x̂)}(b,x̂,l,idx)
end

struct BarycentricValues{T,B,X̂,L,I} <: AbstractVector{T}
    basis::B
    evaluationpoint::X̂
    baryprod::L
    evaluationindex::I
end
BarycentricValues{T}(args...) where {T} = BarycentricValues{T,typeof.(args)...}(args...)

Base.IndexStyle(::Type{<:BarycentricValues}) = Base.IndexLinear()
Base.size(bv::BarycentricValues) = (length(bv.basis),)
function Base.getindex(bv::BarycentricValues, i::Int)
    b = bv.basis
    x = b.interpolationpoints
    λ = b.baryweights
    x̂ = bv.evaluationpoint
    l = bv.baryprod
    idx = bv.evaluationindex

    if idx == 0
        return float(l * λ[i]) / (x̂ - x[i])
    else
        return i == idx ? one(eltype(bv)) : zero(eltype(bv))
    end
end
