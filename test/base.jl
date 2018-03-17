@testset "base" begin

@testset "interpolate" begin
    @testset "1D" begin
        p = interpolate([0],[1]); @test p(1) == 1
        p = interpolate([0],[1],Barycentric()); @test p(1) == 1
        p = interpolate([0],[1],[1]); @test p(1) == 1
        p = interpolate([0],[1],[1],Barycentric()); @test p(1) == 1
        p = interpolate([0],[1],[1],[1]); @test p(1) == 1
        p = interpolate([0],[1],[1],[1],Barycentric()); @test p(1) == 1
    end

    @testset "2D definition" begin
        x = ([0],[0])
        f = reshape([1],(1,1))
        y = ([1],[1])
        p = interpolate(x,f); @test p(1,1) == 1
        p = interpolate(x,f,Barycentric()); @test p(1,1) == 1
        p = interpolate(x,f,y); @test p(1,1) == 1
        p = interpolate(x,f,y,Barycentric()); @test p(1,1) == 1
        p = interpolate(x,f,y,y); @test p(1,1) == 1
        p = interpolate(x,f,y,y,Barycentric()); @test p(1,1) == 1
    end

    @testset "2D evaluation" begin
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
