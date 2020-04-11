"""
    LanczosBasis(n) -> LanczosBasis(n,identity,one,L2(n))
    LanczosBasis(n,a,b,p::InnerProduct)

Orthogonal basis for `span{b(x), a(x)*b(x), ..., a(x)^(n-1)*b(x)}`.
"""
struct LanczosBasis{T,A,B,P} <: AbstractBasis
    length::Int
    a_fun::A
    b_fun::B
    inner_product::P

    # Recursion coefficients
    invnorms::Vector{T}
    dots1::Vector{T}
    dots2::Vector{T}
end

LanczosBasis(n) = LanczosBasis(n,identity,one,L2(n))
function LanczosBasis(n,a_fun,b_fun,p::InnerProduct)
    a = evaluate(a_fun,p)
    b = evaluate(b_fun,p)

    T = typeof(dot(a,b,p))
    invnorms = Vector{T}(undef,n)
    dots1 = Vector{T}(undef,max(0,n-1))
    dots2 = Vector{T}(undef,max(0,n-2))

    n == 0 && return LanczosBasis(n,a_fun,b_fun,p, invnorms,dots1,dots2)
    invnorms[1] = inv(norm(b,p))
    n == 1 && return LanczosBasis(n,a_fun,b_fun,p, invnorms,dots1,dots2)
    q1 = invnorms[1].*b

    q̃2 = a.*q1
    dots1[1] = dot(q̃2,q1,p)
    q̃2 = q̃2 .- dots1[1].*q1
    dots2_end = norm(q̃2,p)
    invnorms[2] = inv(dots2_end)
    q2 = invnorms[2].*q̃2

    for i = 3:n
        dots2[i-2] = dots2_end
        q̃3 = a.*q2
        dots1[i-1] = dot(q̃3,q2,p)
        q̃3 = q̃3 .- dots1[i-1].*q2 .- dots2[i-2].*q1
        dots2_end = norm(q̃3,p)
        invnorms[i] = inv(dots2_end)
        q3 = invnorms[i].*q̃3
        q1,q2 = q2,q3
    end
    return LanczosBasis(n,a_fun,b_fun,p, invnorms,dots1,dots2)
end

Base.length(B::LanczosBasis) = B.length

approxpoints(B::LanczosBasis) = evalpoints(B.inner_product)
approxtransform(B::LanczosBasis,f) = dot(B,f,w)

function iterate_basis(B::LanczosBasis, x)
    length(B) == 0 && return nothing
    q1 = B.invnorms[1]*B.b_fun(x)
    return q1,(2,B.a_fun(x),q1,q1)
end

function iterate_basis(B::LanczosBasis, x, (i,a,q2,q1))
    i > length(B) && return nothing
    if i == 2
        q2 = B.invnorms[i]*(a*q1 - B.dots1[i-1]*q1)
    else
        q1,q2 = q2, B.invnorms[i]*(a*q2 - B.dots1[i-1]*q2 - B.dots2[i-2]*q1)
    end
    return q2,(i+1,a,q2,q1)
end
