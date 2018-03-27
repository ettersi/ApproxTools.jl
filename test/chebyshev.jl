@testset "chebyshev" begin

    @testset "chebpoints" begin

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

    @testset "baryweights" for T in Floats, n = 1:5
        x = chebpoints(T,n)
        s,w = baryweights(x)
        sref,wref = baryweights(collect(x))
        @test s^n*w ≈ sref^n*wref
    end

end
