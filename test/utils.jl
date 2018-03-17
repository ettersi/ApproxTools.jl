@testset "utils" begin

@testset "map2refinterval" for T in (Floats...,complex.(Floats)...)
    a,b = myrand(T,2)
    T = map2refinterval(a,b)
    @test T*a ≈ -1
    @test T*b ≈ 1
    @test T\-1 ≈ a
    @test T\1 ≈ b
end

end
