@testset "fnorm" begin
    @testset "fndims" begin
        using ApproxTools: fndims

        @test fndims(one) == 1
        @test fndims(x->1) == 1
        @test fndims((x,y)->1) == 2
        @test fndims((x,y,z)->1) == 3

        @test fndims(x->1,x->1) == 1
        @test fndims((x,y)->1,(x,y)->1) == 2
        @test fndims((x,y)->1,x->1) == 2
    end

    @test @inferred(fnorm(identity)) == 1
    @test @inferred(fnorm(x->1-x^2, 2)) == 0
    @test @inferred(fnorm(x->1-x^2, (2,))) == 0
    @test @inferred(fnorm(Float32,sin)) == sin(1f0)

    @test @inferred(fnorm((x,y)->x*y)) == 1
    @test @inferred(fnorm((x,y)->1-x^2, 2)) == 0
    @test @inferred(fnorm((x,y)->(1-x^2)*(1.5-y^2), (4,2))) ≈ (1-1/9)*0.5
    @test @inferred(fnorm(Float32,(x,y)->sin(x)*exp(x))) ≈ sin(1f0)*exp(1f0)

    @test @inferred(fnorm(identity, identity)) == 0
    @test @inferred(fnorm(x->1-x^2, x->x^2-1)) == 2
    @test @inferred(fnorm(x->1-x^2, x->x^2-1, 2)) == 0
    @test @inferred(fnorm(x->1-x^2, x->x^2-1, 2)) == 0
    @test @inferred(fnorm(Float32,sin,exp)) == abs(sin(1f0)-exp(1f0))

    @test @inferred(fnorm((x,y)->1,(x,y)->2)) == 1
    @test @inferred(fnorm(Float32,(x,y)->sin(x)*sin(y),(x,y)->exp(x)*sin(y))) == abs(sin(1f0)-exp(1f0))*sin(1f0)
end
