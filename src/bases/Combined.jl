"""
    Combined(n) = [ x-> x^k for k = 0:n-1 ]
"""
struct Combined{B} <: AbstractBasis
    bases::B
end
Combined(bases...) = Combined{typeof(bases)}(bases)

Base.length(B::Combined) = mapreduce(length,+,B.bases)

function iterate_basis(
    B::Combined, x,
    Is = (Iterators.flatten(IterTools.imap(Bi->Bi|x,B.bases)),)
)
    v,s = IterTools.@ifsomething iterate(Is...)
    return v,(Is[1],s)
end

function evaluate_basis(B::Combined,i,x)
    for Bi in B.bases
        i <= length(Bi) && return evaluate_basis(Bi,i,x)
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
