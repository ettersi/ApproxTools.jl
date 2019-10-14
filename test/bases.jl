function test_empty(B)
    @test length(B) == 0
    @test iterate(B|42) == nothing
end

unit(n,i) = [i == j for j = 1:n]

matrix_reference(fun,x) = [fj(xi) for xi in x, fj in fun]
matrix_matrix(B,x) = Matrix(B,x)
matrix_iterate(B,x) = invoke(Matrix, Tuple{ApproxTools.AbstractBasis,AbstractVector}, B,x)
matrix_getindex(B,x) = [B[j](xi) for xi in x, j in 1:length(B)]
matrix_evaltransform_1D(B,x) = hcat((ApproxTools.evaltransform(B,x,unit(length(B),i)) for i = 1:length(B))...)
matrix_evaltransform_2D(B,x) = ApproxTools.evaltransform(B,x,Matrix(I,(length(B),length(B))))

function test_values(B,fun,args...)
    ref = matrix_reference(fun,args...)
    @test matrix_matrix(B,args...) ≈ ref
    @test matrix_iterate(B,args...) ≈ ref
    @test matrix_getindex(B,args...) ≈ ref
    @test matrix_evaltransform_1D(B,args...) ≈ ref
    @test matrix_evaltransform_2D(B,args...) ≈ ref
end

function test_inferred(B,x)
    @inferred(Matrix(B,x))
    @inferred(collect(B|x[1]))
    @inferred(B[1](x[1]))
    @inferred(ApproxTools.evaltransform(B,x,unit(length(B),1)))
    @inferred(ApproxTools.evaltransform(B,x,Matrix(I,(length(B),length(B)))))
end

function test_matrix_eval(B,x)
    ref = Matrix(B,x)
    M = Diagonal(collect(x))
    for (i,Mi) in enumerate(B|M)
        @test diag(Mi) ≈ ref[:,i]
        @test diag(B[i](M)) ≈ ref[:,i]
    end
end

function test_matvec_eval(B,x)
    ref = Matrix(B,x)
    M = Diagonal(collect(x))
    v = ones(length(x))
    for (i,vi) in enumerate(B|(M,v))
        @test vi ≈ ref[:,i]
        @test B[i]((M,v)) ≈ ref[:,i]
    end
end

function test_approximation(B, fun)
    n = length(B)
    for i = 1:n
        @test coeffs(approximate(fun[i],B)) ≈ unit(n,i)
    end
end



@testset "Monomials" begin
    fun = [one,identity,x->x^2,x->x^3,x->x^4]
    B = n->Monomials(n)

    test_empty(B(0))
    test_inferred(B(3), TrapezoidalPoints(3))
    @testset for nB = 1:length(fun)
        @testset for nx = 1:5
            test_values(B(nB), fun[1:nB], TrapezoidalPoints(nx))
            test_matrix_eval(B(nB), TrapezoidalPoints(nx))
            test_matvec_eval(B(nB), TrapezoidalPoints(nx))
        end
        test_approximation(B(nB), fun[1:nB])
    end
end

@testset "NegativeMonomials" begin
    fun = [x->x^-k for k = 1:6]
    B = n->NegativeMonomials(n)

    test_empty(B(0))
    test_inferred(B(4), TrapezoidalPoints(4))
    @testset for nB = 1:length(fun)
        @testset for nx = 2:2:6
            test_values(B(nB), fun[1:nB], TrapezoidalPoints(nx))
            test_matrix_eval(B(nB), TrapezoidalPoints(nx))
            test_matvec_eval(B(nB), TrapezoidalPoints(nx))
        end
    end
end

@testset "Chebyshev" begin
    fun = [one, identity, x->2x^2-1, x->4x^3-3x, x->8x^4-8x^2+1]
    B = n->Chebyshev(n)

    test_empty(B(0))
    test_inferred(B(3), TrapezoidalPoints(3))
    @testset for nB = 1:length(fun)
        @testset for nx = 1:5
            test_values(B(nB), fun[1:nB], TrapezoidalPoints(nx))
            test_matrix_eval(B(nB), TrapezoidalPoints(nx))
            test_matvec_eval(B(nB), TrapezoidalPoints(nx))
        end
        test_approximation(B(nB), fun[1:nB])
    end

    test_inferred(B(3), ChebyshevPoints(3))
    @testset for nB = 1:length(fun)
        @testset for nx = 1:5
            test_values(B(nB), fun[1:nB], ChebyshevPoints(nx))
        end
    end
end

@testset "Weighted" begin
    exp_approx = approximate(exp, Monomials(5))
    monomials = [one,identity,x->x^2,x->x^3,x->x^4]
    fun = [x->exp_approx(x)*p(x) for p in monomials]
    B = n->Weighted(Monomials(n),exp_approx)

    test_empty(B(0))
    test_inferred(B(3), TrapezoidalPoints(3))
    @testset for nB = 1:length(fun)
        @testset for nx = 1:5
            test_values(B(nB), fun[1:nB], TrapezoidalPoints(nx))
            test_matrix_eval(B(nB), TrapezoidalPoints(nx))
            test_matvec_eval(B(nB), TrapezoidalPoints(nx))
        end
    end
end

@testset "Poles" begin
    z = [2,3,4]
    fun = [x->1/(x - z) for z in z]
    B = n->Poles(z[1:n])

    test_empty(B(0))
    test_inferred(B(3), TrapezoidalPoints(3))
    @testset for nB = 1:length(fun)
        @testset for nx = 1:5
            test_values(B(nB), fun[1:nB], TrapezoidalPoints(nx))
            test_matrix_eval(B(nB), TrapezoidalPoints(nx))
            test_matvec_eval(B(nB), TrapezoidalPoints(nx))
        end
    end
end

@testset "Basis" begin
    fun = (exp,sin,cos)
    B = n->Basis(fun[1:n])

    test_empty(B(0))
    @testset for nB = 1:length(fun)
        @testset for nx = 1:5
            test_values(B(nB), fun[1:nB], TrapezoidalPoints(nx))
        end
    end
end

@testset "Combined" begin
    monomials = [one,identity,x->x^2]
    chebyshev = [one, identity, x->2x^2-1]

    test_empty(Combined(Monomials(0),Chebyshev(0)))
    test_inferred(Combined(Monomials(3),Chebyshev(3)), TrapezoidalPoints(3))
    @testset for nx = 1:5
        test_values(Combined(Monomials(3),Chebyshev(3)), [monomials;chebyshev;], TrapezoidalPoints(nx))
        test_values(Combined(Monomials(0),Chebyshev(3)), [          chebyshev;], TrapezoidalPoints(nx))
        test_values(Combined(Monomials(3),Chebyshev(0)), [monomials;          ], TrapezoidalPoints(nx))
    end
end

@testset "Barycentric" begin
    nodepoly = (i,x,y=Float64[]) -> (x̂ -> prod(x̂.-x[[1:i-1;i+1:length(x)]])./prod(x̂.-y))
    fun = (x,y=Float64[])->[x̂->nodepoly(i,x,y)(x̂)/nodepoly(i,x,y)(x[i]) for i = 1:length(x)]

    test_empty(Barycentric(Float64[]))
    test_inferred(Barycentric(TrapezoidalPoints(3)),TrapezoidalPoints(3))
    @testset for Points in (TrapezoidalPoints,ChebyshevPoints)
        @testset for nB = 1:3
            @testset for nx = 1:5
                x = Points(nx)
                test_values(Barycentric(x), fun(x), TrapezoidalPoints(nx))
                @testset for ny = 1:nB-1
                    y = 1 .+ (1:ny)
                    test_values(Barycentric(x,y), fun(x,y), TrapezoidalPoints(nx))
                end
            end
        end
    end
end
