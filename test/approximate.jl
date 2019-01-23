@testset "approximate" begin
    x = [-1,1]
    b = Monomials(2)
    @test coeffs(approximate(one, b)) ≈ [1,0]
    @test coeffs(approximate(identity, b)) ≈ [0,1]
    @test coeffs(approximate(one, (x,b))) ≈ [1,0]
    @test coeffs(approximate(identity, (x,b))) ≈ [0,1]

    @test coeffs(approximate((x1,x2)->x1 - x2, b)) ≈ [0 -1; 1 0]
    @test coeffs(approximate((x1,x2)->x1 - x2, (x,b))) ≈ [0 -1; 1 0]
    @test coeffs(approximate((x1,x2)->x1 - x2, (b,b))) ≈ [0 -1; 1 0]
    @test coeffs(approximate((x1,x2)->x1 - x2, ((x,b),(x,b)))) ≈ [0 -1; 1 0]
end
