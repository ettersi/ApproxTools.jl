struct Midpoints{T <: AbstractFloat} <: AbstractVector{T}
    data::LinRange{T}
    function Midpoints{T}(n) where {T}
        if n == 0
            return new{T}(LinRange{T}(0,0,0))
        else
            return new{T}(LinRange{T}(-1,1,2n+1)[2:2:end-1])
        end
    end
end
Midpoints(n) = Midpoints{Float64}(n)
Base.size(x::Midpoints) = size(x.data)
Base.getindex(x::Midpoints, i::Int) = x.data[i]
