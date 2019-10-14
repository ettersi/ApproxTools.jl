"""
    ArgumentWrapper

`ArgumentWrapper`s standardize the interface between `AbstractBasis` and the
various arguments with which a basis may be evaluated.
"""
abstract type ArgumentWrapper end

"""
    wrap(x)

Wrap `x` in an `ArgumentWrapper`, if needed.

# Examples
```
wrap(x::Number) = x
wrap(x::AbstractMatrix) = MatrixWrapper(x)
```
"""
wrap(x) = x

"""
    val(x)

Obtain the value of the argument `x`.

# Examples
```
val(x::MatrixVectorWrapper) = x.matrix * x.vector
```
"""
val(x) = x

Base.:+(s::UniformScaling, a::ArgumentWrapper) = a + s
Base.:-(a::ArgumentWrapper, s::UniformScaling) = a + (-s)



"""
    MatrixWrapper(M) <: ArgumentWrapper

Wrap the matrix-like object `M` as an argument to `AbstractBasis`.
"""
struct MatrixWrapper{M} <: ArgumentWrapper
    matrix::M
end

wrap(x::AbstractMatrix) = MatrixWrapper(x)

Base.one(a::MatrixWrapper{<:Matrix}) = Matrix{eltype(a.matrix)}(I,size(a.matrix))
Base.one(a::MatrixWrapper{<:Diagonal}) = Diagonal(one.(diag(a.matrix)))
Base.one(a::MatrixWrapper{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = SparseMatrixCSC{Tv,Ti}(I,size(a.matrix))
val(a::MatrixWrapper) = a.matrix

Base.:+(a::MatrixWrapper, s::UniformScaling) = MatrixWrapper(a.matrix + s)
Base.:*(a::MatrixWrapper, b) = a.matrix*b
Base.:\(a::MatrixWrapper, b) = a.matrix\b
Base.inv(a::MatrixWrapper) = inv(a.matrix)
Base.:^(a::MatrixWrapper, p) = a.matrix^p



"""
    MatrixVectorWrapper(M,v, mv_inv = \\) <: ArgumentWrapper

Wrap the matrix-vector pair `M,v` as an argument to `AbstractBasis`.
"""
struct MatrixVectorWrapper{M,V,I} <: ArgumentWrapper
    matrix::M
    vector::V
    mv_inv::I
end

MatrixVectorWrapper(M,v) = MatrixVectorWrapper(M,v,\)

wrap((M,v)::NTuple{2,Any}) = MatrixVectorWrapper(M,v)
Base.one(a::MatrixVectorWrapper) = a.vector
val(a::MatrixVectorWrapper) = a.matrix*a.vector

Base.:+(a::MatrixVectorWrapper, s::UniformScaling) = MatrixVectorWrapper(a.matrix + s, a.vector)
Base.:*(a::MatrixVectorWrapper, b) = a.matrix*b
Base.:\(a::MatrixVectorWrapper, b) = a.matrix\b
Base.inv(a::MatrixVectorWrapper) = a.mv_inv(a.matrix,a.vector)
function Base.:^(a::MatrixVectorWrapper, p::Integer)
    m,v = a.matrix, a.vector
    for i = 1:p
        v = m*v
    end
    for i = 1:-p
        v = m\v
    end
    return v
end
