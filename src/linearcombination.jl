abstract type Basis end

Base.eltype(::Type{B},::Type{X̂}) where {B<:Basis,X̂<:AbstractVector} = eltype(B,eltype(X̂))
Base.eltype(b::Basis,x̂::Union{Number,AbstractVector}) = eltype(typeof(b),typeof(x̂))

function interpolationpoints end

struct LinearCombination{N,C<:AbstractArray{<:Number,N},B<:NTuple{N,Basis}}
    coefficients::C
    basis::B
end
LinearCombination(c::AbstractArray{<:Number,N},b::NTuple{N,Basis}) where {N} = LinearCombination{N,typeof(c),typeof(b)}(c,b)
LinearCombination(c::AbstractArray{<:Number,N},b::Basis) where {N} = LinearCombination(c,(b,))

interpolate(f,b::Basis) = interpolate(f,(b,))
interpolate(f,b::NTuple{N,Basis}) where {N} = LinearCombination(grideval(f, interpolationpoints.(b)), b)

function evaluate(
    c::AbstractVector,
    b::Basis,
    x::Number
)
    @assert length(c) == length(b)
    T = promote_type(eltype(c),eltype(b,x))
    return mapreduce(p -> p[1]*p[2], +, zero(T), zip(c,b(x)))
end

function evaluate(
    c::AbstractArray{<:Any,N},
    b::NTuple{N,Basis},
    x::NTuple{N,Union{Number,AbstractVector}}
) where {N}
    @assert size(c) == length.(b)
    tucker(c, map(b,x) do bk,xk
        T = promote_type(eltype(c), eltype(bk,xk))
        M = Matrix{T}(undef, length(bk),length(xk))
        for j = 1:length(xk)
            copyto!(@view(M[:,j]), bk(xk[j]))
        end
        return transpose(M)
    end)
end

(lc::LinearCombination{N})(x::Vararg{Union{Number,AbstractVector},N}) where {N} = lc(x)
(lc::LinearCombination{N})(x::NTuple{N,Number}) where {N} = evaluate(lc.coefficients, lc.basis, x)[1]
(lc::LinearCombination{N})(x::NTuple{N,Union{Number,AbstractVector}}) where {N} = evaluate(lc.coefficients, lc.basis, x)
(lc::LinearCombination{1})(x::NTuple{1,Number}) = evaluate(lc.coefficients, lc.basis[1], x[1])
(lc::LinearCombination{1})(x::NTuple{1,AbstractVector}) = throw(MethodError(lc,x))
