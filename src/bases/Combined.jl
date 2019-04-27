"""
    Combined(n) = [ x-> x^k for k = 0:n-1 ]
"""
struct Combined{B} <: AbstractBasis
    bases::B
end
Combined(bases...) = Combined{typeof(bases)}(bases)

Base.length(B::Combined) = mapreduce(length,+,B.bases)

Base.:|(B::Combined,x) = Iterators.flatten(IterTools.imap(Bi->Bi|x,B.bases))

function Base.getindex(B::Combined,i)
    for Bi in B.bases
        i <= length(Bi) && return Bi[i]
        i -= length(Bi)
    end
    throw(BoundsError(B,i))
end

function evaltransform(B::Combined, x, c)
    f = Array{typeof(one(eltype(c)) * one(eltype(B|x)))}(undef, (length(x),Base.tail(size(c))...))
    i = 0
    for Bi in B.bases
        idx = ( i+(1:length(Bi)), Base.tail(Base.OneTo.(length.(B.bases)))... )
        @views f[idx...] .= evaltransform(Bi,x,c[idx...])
        i += length(Bi)
    end
    return f
end
