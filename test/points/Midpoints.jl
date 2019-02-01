@testset "Midpoints" begin
    @test_throws Exception Midpoints{Int}(2)
    @test eltype(@inferred(Midpoints(2))) == Float64

    for T in Floats
        @test eltype(@inferred(Midpoints{T}(3))) == T
        @test typeof(@inferred(Midpoints{T}(3)[1])) == T

        @test Midpoints{T}(0) ≈ T[]
        @test Midpoints{T}(1) ≈ T[0]  atol = eps(real(T))
        @test Midpoints{T}(2) ≈ T[-0.5,0.5]
        @test Midpoints{T}(3) ≈ T[-T(2)/3,0,T(2)/3]
        @test Midpoints{T}(4) ≈ T[-0.75,-0.25,0.25,0.75]
    end
end
