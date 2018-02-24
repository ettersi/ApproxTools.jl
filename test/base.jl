@testset "base" begin

@testset "interpolate" begin
    @testset "polynomial" begin
        p = interpolate([0],[1])
        @test p(1) == 1

        p = interpolate(([0],[0]),reshape([1],(1,1)))
        @test p(1,1) == 1
        @test p((1,1)) == 1
        @test p(([1],[1])) == reshape([1],(1,1))
    end
end

@testset "map2refinterval" for T in (Floats...,complex.(Floats)...)
    a,b = myrand(T,2)
    T = map2refinterval(a,b)
    @test T*a ≈ -1
    @test T*b ≈ 1
    @test T\-1 ≈ a
    @test T\1 ≈ b
end

end
