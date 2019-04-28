@testset "TrapezoidalPoints" begin
    @test_throws Exception TrapezoidalPoints{Int}(4)
    @test eltype(@inferred(TrapezoidalPoints(4))) == Float64

    for T in Floats
        @test eltype(@inferred(TrapezoidalPoints{T}(3))) == T
        @test typeof(@inferred(TrapezoidalPoints{T}(3)[1])) == T

        @test TrapezoidalPoints{T}(0) ≈ T[]
        @test TrapezoidalPoints{T}(1) ≈ T[0]  atol = eps(real(T))
        @test TrapezoidalPoints{T}(2) ≈ T[-1,1]
        @test TrapezoidalPoints{T}(3) ≈ T[-1,0,1]   atol = eps(real(T))
        @test TrapezoidalPoints{T}(4) ≈ T[-1,-T(1)/3,T(1)/3,1]
        @test TrapezoidalPoints{T}(5) ≈ T[-1,-T(1)/2,0,T(1)/2,1]   atol = eps(real(T))
    end
end
