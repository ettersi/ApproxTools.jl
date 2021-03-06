@testset "ChebyshevPoints" begin
    @test_throws Exception ChebyshevPoints{Int}(2)
    @test eltype(@inferred(ChebyshevPoints(2))) == Float64

    for T in Floats
        @test eltype(@inferred(ChebyshevPoints{T}(3))) == T
        @test typeof(@inferred(ChebyshevPoints{T}(3)[1])) == T

        @test ChebyshevPoints{T}(0) ≈ T[]
        @test ChebyshevPoints{T}(1) ≈ T[0]  atol = eps(real(T))
        @test ChebyshevPoints{T}(2) ≈ T[-1,1]
        @test ChebyshevPoints{T}(3) ≈ T[-1,0,1]
        @test ChebyshevPoints{T}(4) ≈ T[-1,-0.5,0.5,1]
        @test ChebyshevPoints{T}(5) ≈ T[-1,-1/sqrt(T(2)),0,1/sqrt(T(2)),1]
    end
end
