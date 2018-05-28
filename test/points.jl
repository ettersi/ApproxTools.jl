@testset "points" begin

    @testset "chebpoints" begin
        @test_throws MethodError chebpoints(Int,2)
        @test eltype(@inferred(chebpoints(2))) == Float64

        for T in Floats
            @test eltype(@inferred(chebpoints(T,3))) == T
            @test typeof(@inferred(chebpoints(T,3)[1])) == T

            @test chebpoints(T,0) ≈ T[]
            @test chebpoints(T,1) ≈ T[0]  atol = eps(real(T))
            @test chebpoints(T,2) ≈ T[-1,1]
            @test chebpoints(T,3) ≈ T[-1,0,1]
            @test chebpoints(T,4) ≈ T[-1,-0.5,0.5,1]
            @test chebpoints(T,5) ≈ T[-1,-1/sqrt(T(2)),0,1/sqrt(T(2)),1]
        end
    end

    @testset "prodpot chebpoints" begin
        using ApproxTools: prodpot
        @testset for T in Floats, n = 1:5
            x = chebpoints(T,n)
            w = @inferred(prodpot(x))
            ŵ = prodpot(collect(x))
            @test convert.(float(real(T)),w) ≈ convert.(float(real(T)),ŵ)
        end
    end

    @testset "equipoints" begin
        @testset for n = 1:5
            @test equipoints(n,[]) ≈ chebpoints(n)
        end

        @test equipoints(5,[im]) ≈ [
            -0.9999999999999929,
            -0.682441158315747,
             7.105427357601002e-15,
             0.682441158315747,
             0.9999999999999929
        ]
    end

end
