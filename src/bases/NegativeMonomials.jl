"""
    NegativeMonomials(n) = [ x-> x^-k for k = 1:n ]
"""
struct NegativeMonomials <: AbstractBasis
    n::Int
end

Base.length(B::NegativeMonomials) = B.n

# approxpoints(B::NegativeMonomials) = TODO
# approxtransform(B::NegativeMonomials,f) = TODO

function iterate_basis(B::NegativeMonomials, x)
    length(B) == 0 && return nothing
    p = inv(x)
    return p,(2,p)
end
function iterate_basis(B::NegativeMonomials, x, (i,p))
    i > length(B) && return nothing
    p = x\p
    return p,(i+1,p)
end

evaluate_basis(B::NegativeMonomials,i,x) = x^-i
