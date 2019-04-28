"""
    baryweights(x, y = nothing, csy = nothing) -> w

Compute the barycentric weights for the interpolation points `x`
and poles `[y; csy; conj.(csy)]`.
"""
function baryweights(x, y = nothing)
    n = length(x)

    wx = prodpot(x)
    wy = prodpot(x,y)
    s = LogNumber(true, mean(logabs.(wx)) - mean(logabs.(wy)))
    return float.(s.*wy./wx)
end

"""
    prodpot(x[,y::AbstractVector=x])

Evaluate the product potential `prod(x .- y)` at the scalar `x`,
or evaluate the potential pointwise if `x` is a vector. The result
is returned as a (vector of) `LogNumber`s to avoid over- or
underflow.

Pointwise evaluation is provided as a separate function rather than
through the dot syntax `prodpot.(x,Ref(y))` to allow this function
to be overloaded for point sets where `prodpot(x,y)` can be evaluated
more efficiently (e.g. Chebyshev points).
"""
prodpot(x) = default_prodpot(x)
prodpot(x,y) = default_prodpot(x,y)
prodpot(x,y::Nothing) = true

function default_prodpot(x)
    n = length(x)
    return (i -> prodpot(x[i], @view x[[1:i-1; i+1:n]])).(1:n)
end
default_prodpot(x,y) = default_prodpot.(x,Ref(y))
default_prodpot(x::Number,y) = mapreduce(yi->LogNumber(x - yi), *, y, init=LogNumber(one(x-one(eltype(y)))))

"""
    baryfactor(x̂,x,w) -> î,fac

If `all(x̂.!=x)`, returns `fac = sum(w./(x̂ .- x))` and an undefined `î`.
Otherwise, returns an `î` such that `x̂ == x[î]` and an undefined `fac`.
"""
function baryfactor(x̂,x,w)
    @assert length(x) == length(w)
    n = length(x)
    T = typeof(one(eltype(w))/(x̂-one(eltype(x))))
    fac = zero(T)
    for i = 1:n
        x[i] == x̂ && return i,fac
        fac += w[i]/(x̂-x[i])
    end
    return 0,fac
end

struct Barycentric{X,Y,W} <: AbstractBasis
    points::X
    poles::Y
    weights::W
end

Barycentric(x, y = nothing) = Barycentric(x,y, baryweights(x,y))

Base.length(B::Barycentric) = length(B.points)

approxpoints(B::Barycentric) = B.points
approxtransform(B::Barycentric,f) = f

function iterate_basis(B::Barycentric, x̂, (i,î,fac) = (1,baryfactor(x̂,B.points,B.weights)...))
    i > length(B) && return nothing
    x = B.points
    w = B.weights
    î == 0 && return w[i]/((x̂ - x[i]) * fac), (i+1,î,fac)
    T = typeof(iterate_basis(B,x̂, (i,0,fac))[1])
    return (i == î ? one(T) : zero(T)), (i+1,î,fac)
end

evaluate_basis(B::Barycentric,i,x̂) = iterate_basis(B,x̂,(i,baryfactor(x̂,B.points,B.weights)...))[1]

# function evaltransform(B::Barycentric, x̂::Number, c::AbstractVector)
#     # TODO: implement this operation with single pass over data
# end
