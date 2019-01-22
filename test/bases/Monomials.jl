@testset "Monomial" begin
    b = Monomial(5)
    x = collect(range(-1,stop=1,length=11))
    monos = [one.(x), x, @.(x^2), @.(x^3), @.(x^4)]
    v = rand(length(x))
    @test @inferred(collect(b,x)) ≈ hcat(monos...)
    @test all(@inferred(collect(b(Matrix(Diagonal(x))))) .≈ Diagonal.(monos))
    @test all(@inferred(collect(b(Diagonal(x)))) .≈ Diagonal.(monos))
    @test all(@inferred(collect(b(Diagonal(x),v))) .≈ Diagonal.(monos).*(v,))

    @test all(b[3].(x) .≈ monos[3])
    @test all(b[3](Diagonal(x)) .≈ Diagonal(monos[3]))
end
