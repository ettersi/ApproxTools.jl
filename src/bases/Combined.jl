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
    T = typeof(one(eltype(c)) * one(eltype(B,x[1])))
    f = zeros(T, length(x), Base.tail(size(c))...)
    i = 0
    for Bi in B.bases
        fidx = ( 1:length(x), Base.OneTo.(Base.tail(size(c)))... )
        cidx = ( i.+(1:length(Bi)), Base.OneTo.(Base.tail(size(c)))... )
        if all(length.(cidx) .!= 0)
            @views f[fidx...] .+= evaltransform(Bi,x,c[cidx...])
            i += length(Bi)
        end
    end
    return f
end
