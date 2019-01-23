"""
    module MatFun

Utility functions for abstracting away the differences between
scalar and matrix types.
"""
module MatFun
    using LinearAlgebra

    zero(::Type{T}, x::Number) where {T} = Base.zero(T)
    zero(::Type{T}, x::AbstractMatrix) where {T} = zeros(T,size(x))
    zero(::Type{T}, x::Diagonal) where {T} = Diagonal(Base.zeros(size(x,1)))
    zero(::Type{T}, x::Tuple{AbstractMatrix,AbstractVector}) where {T} = zeros(T,length(x[2]))

    one(x::Number) = Base.one(x)
    one(x::AbstractMatrix) = Matrix(I,size(x))
    one(x::Diagonal) = Diagonal(Base.one.(diag(x)))
    one(x::Tuple{AbstractMatrix,AbstractVector}) = x[2]

    xmul(x::Union{Number,AbstractMatrix}) = x
    xmul(x::Tuple{AbstractMatrix,AbstractVector}) = x[1]

    xval(x::Union{Number,AbstractMatrix}) = x
    xval(x::Tuple{AbstractMatrix,AbstractVector}) = x[1]*x[2]

    function basis_eltype(::Type{M}) where {M <: AbstractMatrix}
        # Make sure matrix type is closed under multiplication
        @assert Base.return_types(*,Tuple{M,M}) == [M]
        return M
    end
    function basis_eltype(::Type{Tuple{M,V}}) where {M<:AbstractMatrix,V<:AbstractVector}
        # Make sure vector type is closed under multiplication with matrix
        @assert Base.return_types(*,Tuple{M,V}) == [V]
        return V
    end
end