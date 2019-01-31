function testbasis(B,fun)
    @testset "evaluation" begin
        x = LinRange(-1,1,11)
        M = Diagonal(collect(x))
        v = ones(length(x))
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

    @testset "approximation" begin
        unit = i->[i == j for j = 1:length(B)]
        x = LinRange(-1,1,length(B))

        for i = 1:length(B)
            if hasmethod(ApproxTools.evaluationpoints, Tuple{typeof(B)})
                @test coeffs(approximate(fun[i],B)) ≈ unit(i)
            end
            @test coeffs(approximate(fun[i],(x,B))) ≈ unit(i)
        end
    end
end

monomials = [one,identity,x->x^2,x->x^3,x->x^4]
chebpolys = [one, identity, x->2x^2-1, x->4x^3-3x, x->8x^4-8x^2+1]
exp_approx = approximate(exp, Monomials(5))

@testset "Monomials" begin testbasis(Monomials(length(monomials)), monomials); end
@testset "Chebyshev" begin testbasis(Chebyshev(length(chebpolys)), chebpolys); end
@testset "Weighted" begin testbasis(Weighted(Monomials(length(monomials)),exp_approx), [x->exp_approx(x)*p(x) for p in monomials]); end
@testset "Poles" begin
    z = [2,3,4]
    testbasis(Poles(z), [x->1/(x - z) for z in z])
end
