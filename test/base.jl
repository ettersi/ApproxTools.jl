@testset "base" begin

@testset "interpolate" begin
    p = interpolate([0],[1])
    @test p(1) == 1

    p = interpolate(([0],[0]),reshape([1],(1,1)))
    @test p(1,1) == 1
    @test p((1,1)) == 1
    @test p(([1],[1])) == reshape([1],(1,1))
end

@testset "map2refinterval" for T in Floats
    T = map2refinterval(T(e),T(π))
    @test T*e ≈ -1
    @test T*π ≈ 1
    @test T\-1 ≈ e
    @test T\1 ≈ π
end

end
