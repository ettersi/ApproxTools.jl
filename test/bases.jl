function test_evaluation(B, fun)
    if length(B) == 0
        @testset "evaluation" begin
            @test length(B) == 0
            @test iterate(B|42) == nothing
        end
    else
        @testset "evaluation, #points = $n" for n = 1:5
            x = n > 1 ? LinRange(-1,1,n) : LinRange(0,0,1)
            M = Diagonal(collect(x))
            v = ones(n)
            vals = (p->p.(x)).(fun)

            @test @inferred(Matrix(B,x)) ≈ hcat(vals...)
            @test all(@inferred(collect(B|M)) .≈ Diagonal.(vals))
            @test all(@inferred(collect(B|(M,v))) .≈ vals)

            for i = 1:length(B)
                @test B[i].(x) ≈ vals[i]
                @test B[i](M) ≈ Diagonal(vals[i])
                @test B[i](M,v) ≈ vals[i]
            end
        end
    end
end

function test_approximation(B, fun)
    length(B) == 0 && return

    n = length(B)
    unit = i->[i == j for j = 1:n]
    x = n > 1 ? LinRange(-1,1,n) : LinRange(0,0,1)

    @testset "approximation, i = $i" for i = 1:n
        @test coeffs(approximate(fun[i],B)) ≈ unit(i)
        @test coeffs(approximate(fun[i],(x,B))) ≈ unit(i)
    end
end

monomials = [one,identity,x->x^2,x->x^3,x->x^4]
chebpolys = [one, identity, x->2x^2-1, x->4x^3-3x, x->8x^4-8x^2+1]
exp_approx = approximate(exp, Monomials(5))

@testset "Monomials" begin
    @testset "degree = $n" for n = 0:length(monomials)
        test_evaluation(Monomials(n), monomials[1:n])
        test_approximation(Monomials(n), monomials[1:n])
    end
end

@testset "Chebyshev" begin
    @testset "degree = $n" for n = 0:length(chebpolys)
        test_evaluation(Chebyshev(n), chebpolys[1:n])
        test_approximation(Chebyshev(n), chebpolys[1:n])
    end
end

@testset "Weighted" begin
    @testset "degree = $n" for n = 0:length(monomials)
        test_evaluation(Weighted(Monomials(n),exp_approx), [x->exp_approx(x)*p(x) for p in monomials[1:n]])
    end
end

z = [2,3,4]
poles = [x->1/(x - z) for z in z]
@testset "Poles" begin
    @testset "degree = $n" for n = 0:length(poles)
        test_evaluation(Poles(z[1:n]), poles[1:n])
    end
end

@testset "Basis" begin
    x = LinRange(-1,1,11)
    @test Matrix(Basis(monomials),x) ≈ [f(x) for x in x, f in monomials]
end

@testset "Combined" begin
    test_evaluation( Combined( Monomials(length(monomials)), Chebyshev(length(chebpolys)) ), [monomials; chebpolys;] )
    test_evaluation( Combined( Monomials(length(monomials)), Chebyshev(0) ), [monomials;] )
    test_evaluation( Combined( Monomials(0), Chebyshev(length(chebpolys)) ), [chebpolys;] )
    test_evaluation( Combined( Monomials(0), Chebyshev(0) ), [] )
end
