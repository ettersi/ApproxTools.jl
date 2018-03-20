@testset "tensor" begin

    @testset "tucker" begin
        @testset for TC in rnc((Int,Float64)), TB in rnc((Int,Float64))
            C = TC[1,2]
            B = (TB[3 4; 5 6],)
            @test @inferred(tucker(C,B)) == B[1]*C
        end

        @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
            C = TC[1 2; 3 4]
            B = (TB1[5 6; 7 8], TB2[9 10; 11 12])
            @test @inferred(tucker(C,B)) == B[1]*C*B[2]'
        end
    end

    @testset "×" begin
        a = [1,2]
        b = [1.,2.]

        @inferred ×(a)
        @test eltype(×(a)) == Tuple{Int}
        @test collect(×(a)) == [(1,),(2,)]

        @inferred ×(a,b)
        @test eltype(×(a,b)) == Tuple{Int,Float64}
        @test collect(×(a,b)) == [(1,1.) (1,2.); (2,1.) (2,2.)]
    end

    @testset "⊗" begin
        a = [1,2]
        b = [1.,2.]

        @inferred ×(a)
        @test eltype(⊗(a)) == Int
        @test collect(⊗(a)) == [1,2]

        @inferred ⊗(a,b)
        @test eltype(⊗(a,b)) == Float64
        @test collect(⊗(a,b)) == [1. 2.; 2. 4.]
    end

end