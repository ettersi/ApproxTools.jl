@testset "Basis" begin
    b = Monomials(3)
    @test collect(b,2) == 2 .^ (0:2)'
    @test collect(b,2:3) == [2 .^ (0:2)'; 3 .^ (0:2)']

    bv = b(2)
    @test length(bv) == length(b)
    @test eltype(bv) == Int
    @test collect(bv) == 2 .^ (0:2)

    Id = Matrix(I,(2,2))
    bf = b[3]
    @test bf(2) == 4
    @test bf(2*Id) == 4*Id
    @test bf(2*Id, [1,1]) == 4*[1,1]
end

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
            @test ndims(p) == ndims(typeof(p)) == 2
            @test @inferred(p( 1,1 )) ≈ sum(C)
            @test @inferred(p((1,1))) ≈ sum(C)
            @test @inferred(p( 1,1:2 )) ≈ [1,1]'*C*[1 1; 1 2]
            @test @inferred(p((1,1:2))) ≈ [1,1]'*C*[1 1; 1 2]
            @test @inferred(p( 1:2,1:2 )) ≈ [1 1; 1 2]'*C*[1 1; 1 2]
            @test @inferred(p((1:2,1:2))) ≈ [1 1; 1 2]'*C*[1 1; 1 2]
        end
    end

    @testset "mat/matvec 1d" begin
        C = rand(5)
        p = LinearCombination(C,Monomials(length(C)))
        x = collect(range(-1,stop=1,length=11))
        @test diag(@inferred(p(Diagonal(x)))) == p.(x)
        @test @inferred(p(Diagonal(x),one.(x))) == p.(x)
    end
end
