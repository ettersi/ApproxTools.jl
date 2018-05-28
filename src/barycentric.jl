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
logabs(x::Number) = log(abs(x))
logabs(x::LogNumber) = x.logabs

lognumber(::Type{T}) where {T<:Number} = LogNumber{float(T), float(real(T))}
lognumber(x::Number) = convert(LogNumber, x)
Base.float(x::LogNumber{S,L}) where {S,L} = convert(promote_type(S,L),x)

Base.one(::Type{LogNumber{S,L}}) where {S,L} = LogNumber{S,L}(1,0)

Base.convert(::Type{LogNumber}, x::LogNumber) = x
Base.convert(::Type{LogNumber{S,L}}, x::LogNumber) where {S,L} =
    LogNumber{S,L}(x.sign,x.logabs)
Base.convert(::Type{LogNumber}, x::Number) = LogNumber(float(sign(x)),logabs(x))
Base.convert(::Type{LogNumber{S,L}}, x::Number) where {S,L} =
    LogNumber{S,L}(sign(x),logabs(x))
Base.convert(::Type{T}, x::LogNumber) where {T<:Number} =
    convert(T,x.sign)*exp(convert(real(T),x.logabs))

Base.promote_rule(::Type{LogNumber{S,L}}, ::Type{T}) where {S,L,T<:Number}=
    LogNumber{promote_type(S,T), promote_type(L,float(real(T)))}
Base.promote_rule(::Type{LogNumber{S1,L1}}, ::Type{LogNumber{S2,L2}}) where {S1,L1,S2,L2}=
    LogNumber{promote_type(S1,S2), promote_type(L1,L2)}

Base.:*(x::LogNumber,y::LogNumber) = LogNumber(x.sign * y.sign, x.logabs + y.logabs)
Base.:/(x::LogNumber,y::LogNumber) = LogNumber(x.sign * conj(y.sign), x.logabs - y.logabs)


"""
    prodpot(x)

Efficient and stable implementation of

    [ prod(x[i] - x[setdiff(1:length(x),i)]) for i = 1:length(x) ]

The result is returned as a `Vector{LogNumber}` to avoid over- or
underflow.
"""
function prodpot(x::AbstractVector)
    n = length(x)
    return (i -> mapreduce(
            xj->x[i]-xj, *,
            one(lognumber(float(eltype(x)))),
            @view(x[[1:i-1;i+1:n]])
    )).(1:n)
end

"""
    extract_scale(x) -> s,x̃

Rescale `x` such that `mean(log(x̃)) ≈ 0`.

`s` and `x̃` are defined through the following relations:
 - `x ≈ s^n * x̃`,
 - `mean(log(x̃)) ≈ 0`.

A product of `n` numbers generally scales exponentially in `n` and
hence the result is likely to over- or underflow in most floating-point
types. Rescaling in the above way returns `s` and `x̃` to `O(1)` such
that they can be safely represented in standard floating-point types.
"""
function extract_scale(x::AbstractVector)
    n = length(x)
    logS = mapreduce(logabs, +, x)/n
    return exp(logS/n), (xi->sign(xi)*exp(logabs(xi) - logS)).(x)
end


struct Barycentric{X,P,S,W} <: Basis
    points::X
    potential::P
    scaling::S
    weights::W
end

"""
    Barycentric(x, pot = one)

Basis functions for barycentric interpolation.

The basis functions are given by

   b[k](x̂) = prod(x̂ - x̃) / prod(x[k] - x̃) * pot(x̂) / pot(x[k])

where

    x̃ = x[setdiff(1:n,k)]
"""
Barycentric(x::AbstractVector, pot = one) =
    Barycentric(x,pot, extract_scale(1 ./ (pot.(x) .* prodpot(x)))...)

Base.length(b::Barycentric) = length(b.points)
Base.eltype(::Type{Barycentric{X,P,S,W}},::Type{X̂}) where {X,P,S,W,X̂<:Number} = promote_type(eltype(W),X̂)
interpolationpoints(b::Barycentric) = b.points


function (b::Barycentric)(x̂::Number)
    x = b.points
    pot = b.potential
    s = b.scaling
    l = pot(x̂) * mapreduce(xi->s*(x̂-xi), *, one(promote_type(eltype.((s,x̂,x))...)), x)
    return BarycentricValues(b,x̂,l,findfirst(x,x̂))
end

struct BarycentricValues{B,X,L,I}
    basis::B
    point::X
    l::L
    idx::I
end

Base.length(bv::BarycentricValues) = length(bv.basis)
Base.eltype(bv::BarycentricValues) = eltype(bv.basis,bv.point)
Base.start(bv::BarycentricValues) = 1
function Base.next(bv::BarycentricValues, i)
    b = bv.basis
    x = b.points
    w = b.weights
    x̂ = bv.point
    l = bv.l
    idx = bv.idx

    if idx == 0
        bi = l * w[i] / (x̂ - x[i])
    else
        bi = i == idx ? one(eltype(bv)) : zero(eltype(bv))
    end
    return bi, i+1
end
Base.done(bv::BarycentricValues, i) = i > length(bv)
