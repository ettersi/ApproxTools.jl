"""
    Monomials(n) = [ x-> x^k for k = 0:n-1 ]
"""
struct Monomials <: Basis
    n::Int
end

Base.length(B::Monomials) = B.n

evaluationpoints(B::Monomials) = exp.(2π*im/length(B).*(0:length(B)-1))
approxtransform(B::Monomials,f) = fft(convert(Array,f),1)./length(B)

function iterate_basis(B::Monomials, x)
    p = one(x)
    return p,(2,p)
end
function iterate_basis(B::Monomials, x, (i,p))
    i > length(B) && return nothing
    p = x*p
    return p,(i+1,p)
end

evaluate_basis(B::Monomials,i,x) = x^(i-1)
