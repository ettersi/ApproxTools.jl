@testset "approximate" begin
    # Only test two-dimensional approximation
    # One-dimensional approximation gets tested in bases.jl
    @test coeffs(approximate((x1,x2)->x1*x2, Monomials(2))) ≈ [0 0; 0 1]
end
