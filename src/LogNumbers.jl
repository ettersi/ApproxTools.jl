"""
    LogNumber{S,L}

Represent a real or complex number in terms of its sign and logarithm.

This is a hack around the limitation that standard floating-point
types link the bitcount of the exponent to that of the mantissa, which
makes it impossible to represent very large or small values without
simultaneously increasing the precision of the representation. The
drawback of this hack is that values x close to 1 have a larger relative
accuracy than values far away from 1 since

    1 - exp(l*(1±ε)) / exp(l) == 1 - exp(±l*ε) ≈ ∓l*ε.

It remains to be seen whether this has any practical relevance.

A better workaround would be to create a floating-point type with
arbitrary exponent and mantissa.
"""
struct LogNumber{S,L} <: Number
    sign::S
    logabs::L
end

LogNumber(x::Number) where {S,L} = LogNumber(sign(x),logabs(x))
LogNumber{S,L}(x::Number) where {S,L} = LogNumber{S,L}(sign(x),logabs(x))
Base.AbstractFloat(x::LogNumber) = x.sign*exp(x.logabs)

Base.sign(x::LogNumber) = x.sign

"""
    logabs(x) -> log(abs(x))
"""
logabs(x::Number) = log(abs2(x))/2
logabs(x::LogNumber) = x.logabs

Base.promote_rule(::Type{LogNumber{S,L}}, ::Type{T}) where {S,L,T<:Number} =
    LogNumber{promote_type(S,typeof(sign(one(T)))), promote_type(L,typeof(logabs(one(T))))}

Base.:*(x::LogNumber,y::LogNumber) = LogNumber(x.sign * y.sign, x.logabs + y.logabs)
Base.:/(x::LogNumber,y::LogNumber) = LogNumber(x.sign * conj(y.sign), x.logabs - y.logabs)
