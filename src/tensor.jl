function tucker(
    C::AbstractArray{<:Any,N},
    B::NTuple{N,AbstractMatrix}
) where {N}
    for k = 1:N
        tmp = reshape(C,(size(C,1),prod(Base.tail(size(C)))))
        tmp = B[k]*tmp
        C = convert(Array,reshape(tmp',(Base.tail(size(C))...,size(B[k],1))))
        # ^ Need to explicitely evaluate here, otherwise type inference gets confused
    end
    return C
end

"""
    ×(x...)

TODO
"""
×(x::AbstractVector...) = TensorGrid{Tuple{eltype.(x)...},length(x),typeof(x)}(x)
struct TensorGrid{T,N,F} <: AbstractArray{T,N}
    factors::F
end
Base.size(x::TensorGrid) = length.(x.factors)
Base.getindex(x::TensorGrid{<:Any,N,<:Any}, i::Vararg{Int,N}) where {N} = map(getindex,x.factors,i)

"""
    ⊗(x...)

TODO
"""
function ⊗(x::AbstractVector...)
    grid = ×(x...)
    return TensorProduct{promote_type(eltype.(x)...),length(x), typeof(grid)}(grid)
end
struct TensorProduct{T,N,G} <: AbstractArray{T,N}
    grid::G
end
Base.size(x::TensorProduct) = size(x.grid)
Base.getindex(x::TensorProduct{<:Any,N,<:Any}, i::Vararg{Int,N}) where {N} = prod(x.grid[i...])
