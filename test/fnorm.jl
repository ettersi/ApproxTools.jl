@testset "fnorm" begin
    @testset "fndims" begin
        using ApproxTools: fndims

        foo(::Int) = nothing
        @test fndims(foo) == 0
        @test fndims(x->1) == 1
        @test fndims((x,y)->1) == 2
        @test fndims((x,y,z)->1) == 3

        @test fndims(x->1,x->1) == 1
        @test fndims((x,y)->1,(x,y)->1) == 2
        @test fndims((x,y)->1,x->1) == 2
    end

    p = interpolate(identity, Chebyshev(4))
    @test @inferred(fnorm(p)) == 1
    @test @inferred(fnorm(identity)) == 1
    @test @inferred(fnorm(x->1-x^2, 2)) == 0
    @test @inferred(fnorm(x->1-x^2, (2,))) == 0
    @test @inferred(fnorm(Float32,sin)) == sin(1f0)

    p = interpolate(*, ntuple(k->Chebyshev(4), Val(2)))
    @test @inferred(fnorm(p)) == 1
    @test @inferred(fnorm((x,y)->x*y)) == 1
    @test @inferred(fnorm((x,y)->1-x^2, 2)) == 0
    @test @inferred(fnorm((x,y)->(1-x^2)*(1.5-y^2), (4,2))) ≈ (1-1/9)*0.5
    @test @inferred(fnorm(Float32,(x,y)->sin(x)*exp(x))) ≈ sin(1f0)*exp(1f0)

    p = interpolate(identity, Chebyshev(4))
    @test @inferred(fnorm(p, p)) == 0
    @test abs(@inferred(fnorm(identity, p))) < 10*eps()
    @test @inferred(fnorm(identity, identity)) == 0
    @test @inferred(fnorm(x->1-x^2, x->x^2-1)) == 2
    @test @inferred(fnorm(x->1-x^2, x->x^2-1, 2)) == 0
    @test @inferred(fnorm(x->1-x^2, x->x^2-1, 2)) == 0
    @test @inferred(fnorm(Float32,sin,exp)) == abs(sin(1f0)-exp(1f0))

    p = interpolate(*, ntuple(k->Chebyshev(4),Val(2)))
    @test @inferred(fnorm(p, p)) == 0
    @test abs(@inferred(fnorm(*, p))) < 10*eps()
    @test @inferred(fnorm((x,y)->1,(x,y)->2)) == 1
    @test @inferred(fnorm(Float32,(x,y)->sin(x)*sin(y),(x,y)->exp(x)*sin(y))) == abs(sin(1f0)-exp(1f0))*sin(1f0)
end
