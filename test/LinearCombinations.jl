@testset "LinearCombination" begin
    @testset "scalar 1d" begin
        @testset for TC in rnc((Int,Float64))
            C = rand(TC,3)

            p = @inferred(LinearCombination(C, Monomials(3)))
            @test ApproxTools.GridevalStyle(p) == ApproxTools.GridevalCartesian()
            @test ndims(p) == ndims(typeof(p)) == 1

            @test @inferred(p( 1  )) ≈ sum(C)
            @test @inferred(p((1,))) ≈ sum(C)
            @test @inferred(p( [1,2]  )) ≈ [sum(C), sum(C.*2 .^ (0:2))]
            @test @inferred(p(([1,2],))) ≈ [sum(C), sum(C.*2 .^ (0:2))]
        end
    end

    @testset "scalar 2d" begin
        @testset for TC in rnc((Int,Float64))
            C = rand(TC,2,2)

            p = @inferred(LinearCombination(C, Monomials.((2,2))))
            @test @inferred(LinearCombination(C, Monomials(2))) == p
            @test ApproxTools.GridevalStyle(p) == ApproxTools.GridevalCartesian()
            @test ndims(p) == ndims(typeof(p)) == 2

            @test @inferred(p( 1,1 )) ≈ sum(C)
            @test @inferred(p((1,1))) ≈ sum(C)
            @test @inferred(p( 1,1:2 )) ≈ [1,1]'*C*[1 1; 1 2]
            @test @inferred(p((1,1:2))) ≈ [1,1]'*C*[1 1; 1 2]
            @test @inferred(p( 1:2,1:2 )) ≈ [1 1; 1 2]'*C*[1 1; 1 2]
            @test @inferred(p((1:2,1:2))) ≈ [1 1; 1 2]'*C*[1 1; 1 2]
        end
    end

    @testset "matrix 1d" begin
        C = rand(5)
        p = LinearCombination(C,Monomials(length(C)))
        x = LinRange(-1,1,11)
        M = Diagonal(collect(x))
        @test diag(@inferred(p(M))) == p.(x)
    end

    @testset "matrix times vector 1d" begin
        C = rand(5)
        p = LinearCombination(C,Monomials(length(C)))
        x = LinRange(-1,1,11)
        M = Diagonal(collect(x))
        v = one.(x)
        @test @inferred(((M,v)->@evaluate p(M) * v)(M,v)) == p.(x)
    end
end
