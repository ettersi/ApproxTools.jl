@testset "Chebyshev" begin

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

    @testset "Chebyshev" begin
        b = Chebyshev(5)
        x = collect(LinRange(-1,1,11))
        chebpolys = [one, identity, x->2x^2-1, x->4x^3-3x, x->8x^4-8x^2+1]
        chebvals = (p->p.(x)).(chebpolys)
        v = rand(length(x))
        @test @inferred(collect(b,x)) ≈ hcat(chebvals...)
        @test @inferred(collect(b(Matrix(Diagonal(x))))) ≈ Diagonal.(chebvals)
        @test @inferred(collect(b(Diagonal(x)))) ≈ Diagonal.(chebvals)
        @test @inferred(collect(b(Diagonal(x),v))) ≈ Diagonal.(chebvals).*(v,)

        @test coeffs(@inferred(approximate(chebpolys[1], b))) ≈ [1,0,0,0,0]
        @test coeffs(@inferred(approximate(chebpolys[2], b))) ≈ [0,1,0,0,0]
        @test coeffs(@inferred(approximate(chebpolys[3], b))) ≈ [0,0,1,0,0]
        @test coeffs(@inferred(approximate(chebpolys[4], b))) ≈ [0,0,0,1,0]
        @test coeffs(@inferred(approximate(chebpolys[5], b))) ≈ [0,0,0,0,1]
    end

end
