"""
    rsmo(x)

Evaluate `√(x^2-1)` with branch cut along `[-1,1]`.

The function name is the abbreviation of "root of square minus one".
"""
rsmo(x) = rsmo(float(x))
rsmo(x::Union{T,Complex{T}}) where {T<:AbstractFloat} = ifelse(!signbit(real(x)),1,-1)*sqrt(x^2-1)

"""
    jouk(z)

Joukowsky map `(z+z^-1)/2`.
"""
jouk(z) = (z+inv(z))/2

"""
    ijouk(x; halfplane=Val(false), branch=Val(true))

Inverse Joukowsky map `x ± √(x^2-1)`. The result `z` is
determined as follows.

+-----------+---------------+----------------+
|           | `!halfplane`  |  `halfplane`   |
+-----------+---------------+----------------+
| ` branch` | `abs(z) >= 1` | `imag(z) >= 0` |
| `!branch` | `abs(z) <= 1` | `imag(z) <= 0` |
+-----------+---------------+----------------+

"""
ijouk(z; halfplane=Val(false), branch=Val(true)) = ijouk(z,halfplane,branch)
ijouk(z,::Val{false},::Val{true} ) = z + rsmo(z)
ijouk(z,::Val{false},::Val{false}) = z - rsmo(z)
ijouk(z,::Val{true} ,::Val{true} ) = z + im*sqrt(1-z^2)
ijouk(z,::Val{true} ,::Val{false}) = z - im*sqrt(1-z^2)


"""
   semimajor(z; kwargs...)

Length of semi-major axis of Bernstein ellipse through `z`.

See [`ijouk`](@ref) regardings `kwargs`.
"""
semimajor(z; kwargs...) = (w = abs(ijouk(z; kwargs...)); (w+inv(w))/2)

"""
   semiminor(z; kwargs...)

Length of semi-minor axis of Bernstein ellipse through `z`.

See [`ijouk`](@ref) regardings `kwargs`.
"""
semiminor(z; kwargs...) = (w = abs(ijouk(z; kwargs...)); (w-inv(w))/2)

"""
   radius(z; kwargs...) = abs(ijouk(z; kwargs...))
"""
radius(z; kwargs...) = abs(ijouk(z; kwargs...))
