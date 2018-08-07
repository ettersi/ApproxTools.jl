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
   radius(z)

Radius `r > 1` of the circle `C` such that `jouk.(C)` passes
through `z`.
"""
radius(z) = abs(ijouk(z))
