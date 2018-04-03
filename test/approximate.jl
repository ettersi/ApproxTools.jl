struct MockApproximation <: ApproxTools.ApproximationAlgorithm
end

ApproxTools.approximate(
    T::Type,
    f,
    n::NTuple{<:Any,Integer},
    alg::MockApproximation
) = T,f,n

@testset "approximate" begin
    f = x->x
    n = 1

    @test approximate(f,n,MockApproximation()) == (Float64,f,(n,))
    @test approximate(Int,f,n,MockApproximation()) == (Int,f,(n,))
    @test approximate(f,(n,),MockApproximation()) == (Float64,f,(n,))
    @test approximate(Int,f,(n,),MockApproximation()) == (Int,f,(n,))
end
