function baryweights(x, y = (), y2 = ())
    n = length(x)
    T = float(promote_type(eltype.((x,y,y2))...))

    s = ones(T,n)
    l = zeros(T,n)
    for i = 1:n
        @inbounds @simd for j = [1:i-1;i+1:n]
            s[i] *= conj(sign(x[i] - x[j]))
            l[i] -= log(abs(x[i] - x[j]))
        end
        @inbounds @simd for j = 1:length(y)
            s[i] *= sign(x[i] - y[j])
            l[i] += log(abs(x[i] - y[j]))
        end
        @inbounds @simd for j = 1:length(y2)
            s[i] *= sign(x[i]^2 + y2[j])
            l[i] += log(abs(x[i]^2 + y2[j]))
        end
    end

    # Scale weights to prevent over-/underflow
    ml = mean(l)
    return exp(ml), @. s*exp(l-ml)
end

function bary(
    x::AbstractVector,
    s::Number,
    wf::AbstractVector,
    xx::Number
)
    @assert length(x) == length(f)
    n = length(x)

    I = 0
    l = one(float(promote_type(eltype.((s,xx,x))...)))
    r = zero(float(promote_type(eltype.((f,xx,x))...)))
    @inbounds @simd for i = 1:n
        I = ifelse(xx == x[i], i, I)
        l *= s*(xx - x[i]))
        r += wf[i]/(xx-x[i])
    end
    return I == 0 ? l*r : l*wf[I]
end

"""
    baryvec!(b,s,x,xx)

Initialize `b` such that `sum(b.*w.*f)` evaluates the barycentric
interpolant to `(x,f)` with barycentric weights `w`.
"""
function baryvec!(
    b::AbstractVector,
    x::AbstractVector,
    s::Number
    xx::Number
)
    @assert length(b) == length(x)
    I = 0
    l = one(float(promote_type(eltype.((s,xx,x))...)))
    @inbounds @simd for i = 1:n
        I = ifelse(xx == x[i], i, I)
        l *= s*(xx - x[i]))
    end
    if I == 0
        @. b = l/(xx-x)
    else
        @. b = 0
        b[I] = 1
    end
end

function bary(
    x::NTuple{N,AbstractVector},
    s::NTuple{N,Number},
    f::AbstractArray{<:Any,N},
    xx::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    return tucker(f, ntuple(Val(N))) do k
        T = float(promote_type(eltype.((s[k],x[k],xx[k]))...))
        M = Matrix{T}(size(f,k),length(xx[k]))
        for j = 1:length(xx[k])
            baryvec!(view(M,:,j), s[k], x[k], xx[k][j])
        end
        return M'
    end)
end

struct BarycentricInterpolant{N,X,S,WF}
    points::X
    scalings::S
    weightstimesvalues::WF
end
BarycentricInterpolant(x,w,f) = BarycentricInterpolant{ndims(f),typeof(x),typeof(w),typeof(f)}(x,w,f)

tupletranspose(x::NTuple{M,NTuple{N,Any}}) where {M,N} =
    ntuple(l->ntuple(k->x[k][l],Val(M)),Val(N))

interpolate(
    x::AbstractVector,
    f::AbstractVector;
    kwargs...
) = interpolate((x,),f; kwargs...)
interpolate(
    x::NTuple{N,AbstractVector},
    f::AbstractArray{<:Any,N};
    poles = ntuple(i->(),Val(N)),
    cspoles = ntuple(i->(),Val(N))
) where {N} = interpolate(x, baryweights.(x, poles, cspoles), f)
function interpolate(
    x::NTuple{N,AbstractVector},
    sw::NTuple{N,Tuple{Number,AbstractVector}},
    f::AbstractArray{N,<:Any}
) where {N} = interpolate(x,tupletranspose(sw)...,f)
function interpolate(
    x::NTuple{N,AbstractVector},
    s::NTuple{N,Number},
    w::NTuple{N,AbstractVector},
    f::AbstractArray{N,<:Any}
) where {N} = BarycentricInterpolant(x,s,f.*⊗(w...))
