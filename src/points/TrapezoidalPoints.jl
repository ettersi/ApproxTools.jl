struct TrapezoidalPoints{T <: AbstractFloat} <: AbstractVector{T}
    data::LinRange{T}
    function TrapezoidalPoints{T}(n) where {T}
        if n <= 1
            return new{T}(LinRange{T}(0,0,n))
        else
            return new{T}(LinRange{T}(-1,1,n))
        end
    end
end
TrapezoidalPoints(n) = TrapezoidalPoints{Float64}(n)
Base.size(x::TrapezoidalPoints) = size(x.data)
Base.getindex(x::TrapezoidalPoints, i::Int) = x.data[i]
