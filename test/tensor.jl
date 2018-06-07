@testset "tensor" begin

    @testset "tucker" begin
        using ApproxTools: tucker
        @testset for TC in rnc((Int,Float64)), TB in rnc((Int,Float64))
            C = rand(TC,2)
            B = (rand(TB,2,2),)
            @test @inferred(tucker(C,map(B->(C->B*C),B))) ≈ B[1]*C
        end

        @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
            C = rand(TC,2,2)
            B = (rand(TB1,2,2), rand(TB2,2,2))
            @test @inferred(tucker(C,map(B->(C->B*C),B))) ≈ B[1]*C*transpose(B[2])
        end
    end

    @testset "cartesian" begin
        a = [1,2]
        b = [1.,2.]

        @inferred cartesian( a  )
        @inferred cartesian((a,))
        @test eltype(cartesian( a  )) == Tuple{Int}
        @test eltype(cartesian((a,))) == Tuple{Int}
        @test collect(cartesian( a  )) == [(1,),(2,)]
        @test collect(cartesian((a,))) == [(1,),(2,)]

        @inferred cartesian( a,b )
        @inferred cartesian((a,b))
        @test eltype(cartesian( a,b )) == Tuple{Int,Float64}
        @test eltype(cartesian((a,b))) == Tuple{Int,Float64}
        @test collect(cartesian( a,b )) == [(1,1.) (1,2.); (2,1.) (2,2.)]
        @test collect(cartesian((a,b))) == [(1,1.) (1,2.); (2,1.) (2,2.)]
    end

    @testset "grideval" begin
        a = [1,2]
        b = [1.,2.]

        @inferred grideval(*,  a )
        @inferred grideval(*, (a,))
        @test eltype(grideval(*,  a  )) == Int
        @test eltype(grideval(*, (a,))) == Int
        @test collect(grideval(*,  a  )) == [1,2]
        @test collect(grideval(*, (a,))) == [1,2]

        @inferred grideval(*,  a,b )
        @inferred grideval(*, (a,b))
        @test eltype(grideval(*,  a,b )) == Float64
        @test eltype(grideval(*, (a,b))) == Float64
        @test collect(grideval(*,  a,b )) == [1. 2.; 2. 4.]
        @test collect(grideval(*, (a,b))) == [1. 2.; 2. 4.]

        p = LinearCombination([0,1], Chebyshev(2))
        @inferred grideval(p,  a )
        @inferred grideval(p, (a,))
        @test eltype(grideval(p,  a  )) == Int
        @test eltype(grideval(p, (a,))) == Int
        @test collect(grideval(p,  a  )) == [1,2]
        @test collect(grideval(p, (a,))) == [1,2]

        p = LinearCombination([0 0; 0 1], (Chebyshev(2),Chebyshev(2)))
        @inferred grideval(p,  a,b )
        @inferred grideval(p, (a,b))
        @test eltype(grideval(p,  a,b )) == Float64
        @test eltype(grideval(p, (a,b))) == Float64
        @test collect(grideval(p,  a,b )) == [1. 2.; 2. 4.]
        @test collect(grideval(p, (a,b))) == [1. 2.; 2. 4.]
    end

end
