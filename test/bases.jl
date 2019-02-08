function testbasis(Bgen,fun)
    @testset "evaluation" begin
        for n = 1:length(fun)
            B = Bgen(n)
            for m = 1:5
                x = m > 1 ? LinRange(-1,1,m) : LinRange(0,0,1)
                M = Diagonal(collect(x))
                v = ones(m)
                vals = (p->p.(x)).(fun[1:n])

                @test @inferred(Matrix(B,x)) ≈ hcat(vals...)
                @test all(@inferred(collect(B|M)) .≈ Diagonal.(vals))
                @test all(@inferred(collect(B|(M,v))) .≈ vals)

                for i = 1:n
                    @test B[i].(x) ≈ vals[i]
                    @test B[i](M) ≈ Diagonal(vals[i])
                    @test B[i](M,v) ≈ vals[i]
                end
            end
        end
    end

    @testset "approximation" begin
        for n = 1:length(fun)
            B = Bgen(n)
            unit = i->[i == j for j = 1:n]
            x = n > 1 ? LinRange(-1,1,n) : LinRange(0,0,1)

            for i = 1:n
                if hasmethod(ApproxTools.evaluationpoints, Tuple{typeof(B)})
                    @test coeffs(approximate(fun[i],B)) ≈ unit(i)
                end
                @test coeffs(approximate(fun[i],(x,B))) ≈ unit(i)
            end
        end
    end
end

monomials = [one,identity,x->x^2,x->x^3,x->x^4]
chebpolys = [one, identity, x->2x^2-1, x->4x^3-3x, x->8x^4-8x^2+1]
exp_approx = approximate(exp, Monomials(5))

@testset "Monomials" begin testbasis(n->Monomials(n), monomials); end
@testset "Chebyshev" begin testbasis(n->Chebyshev(n), chebpolys); end
@testset "Weighted" begin testbasis(n->Weighted(Monomials(n),exp_approx), [x->exp_approx(x)*p(x) for p in monomials]); end
@testset "Poles" begin
    z = [2,3,4]
    testbasis(n->Poles(z[1:n]), [x->1/(x - z) for z in z])
end
