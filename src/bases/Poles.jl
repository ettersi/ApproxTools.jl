"""
    Poles(z) = [x->1/(x- z) for z in z]
"""
struct Poles{Z} <: Basis
    poles::Z
end

Base.length(B::Poles) = length(B.poles)

evaluate_basis(B::Poles, i, x) = inv(x - B.poles[i]*I)
