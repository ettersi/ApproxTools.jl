"""
    rsmo(z)

Evaluate `√(z^2-1)` with branch cut along `[-1,1]`.

The function name is the abbreviation of "root of square minus one".
"""
rsmo(z) = rsmo(float(z))
rsmo(z::Union{T,Complex{T}}) where {T<:AbstractFloat} = ifelse(!signbit(real(z)),1,-1)*sqrt(z^2-1)

"""
    jouk(z)

Joukowsky map `(z+z^-1)/2`.
"""
jouk(z) = (z+inv(z))/2

"""
    ijouk(z, branch=Val(true))

Inverse Joukowsky map `z ± √(z^2-1)`. If `branch == Val(true)`,
the sign is chosen such that the result lies outside the unit
circle, and inside the unit circle for `branch == Val{false}`.
This function has a branch cut along `[-1,1]`.
"""
ijouk(z) = ijouk(z,Val(true))
ijouk(z,::Val{true}) = z + rsmo(z)
ijouk(z,::Val{false}) = z - rsmo(z)

"""
    ijoukt(z, branch=Val(true))

Inverse Joukowsky map `z ± im*√(1-z^2)`. If `branch == Val(true)`,
the sign is chosen such that the result lies in the upper half
plane, and in the lower half plane for `branch == Val(false)`.
This function has a branch cut along `Reals \ (-1,1)`.
"""
ijoukt(z) = ijoukt(z,Val(true))
ijoukt(z,::Val{true}) = z + im*sqrt(1-z^2)
ijoukt(z,::Val{false}) = z - im*sqrt(1-z^2)

"""
   semimajor(z)

Length of semi-major axis of Bernstein ellipse through `z`.
"""
semimajor(z) = (w = abs(ijouk(z)); (w+inv(w))/2)

"""
   semiminor(z)

Length of semi-minor axis of Bernstein ellipse through `z`.
"""
semiminor(z) = (w = abs(ijouk(z)); (w-inv(w))/2)

"""
   radius(z) = abs(ijouk(z))
"""
radius(z) = abs(ijouk(z))

"""
   radiust(z) = abs(ijoukt(z))
"""
radiust(z) = abs(ijoukt(z))
