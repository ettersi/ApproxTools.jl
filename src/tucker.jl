apply(a,b) = apply(a,b,Val(hasmethod(*,Tuple{typeof(a),typeof(b)})))
apply(a,b,::Val{true}) = a*b
apply(a,b,::Val{false}) = a(b)

"""
    tucker(C,B)

Apply `B[k]` to the `k`th dimension of tensor `C`.

# Examples
```
julia> C = rand(2); B = (rand(3,2),);
       tucker(C,B) == B[1]*C
true

julia> C = rand(2,3); B = (rand(4,2),rand(5,3));
       tucker(C,B) == B[1]*C*B[2]'
true
```
"""
function tucker end

@generated function tucker(
    C::AbstractArray{<:Any,N},
    B::NTuple{N,Any}
) where {N}
    quote
        C_0 = C
        Base.Cartesian.@nexprs $N k->begin
            tmp_k = reshape(C_{k-1},(size(C_{k-1},1),prod(Base.tail(size(C_{k-1})))))
            tmp_k = apply(B[k],tmp_k)
            C_k = Array(reshape(transpose(tmp_k),(Base.tail(size(C_{k-1}))...,size(tmp_k,1))))
            # https://github.com/JuliaLang/julia/issues/30988
        end
        return $(Symbol("C_",N))
    end
end
tucker(C::AbstractArray{<:Any,1}, B::NTuple{1,Any}) = apply(B[1],C)
tucker(C::AbstractArray{<:Any,2}, B::NTuple{2,Any}) = transpose(apply(B[2],transpose(apply(B[1],C))))
