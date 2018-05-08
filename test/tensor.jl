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

    @testset "tensor" begin
        a = [1,2]
        b = [1.,2.]

        @inferred tensor( a )
        @inferred tensor((a,))
        @test eltype(tensor( a  )) == Int
        @test eltype(tensor((a,))) == Int
        @test collect(tensor( a  )) == [1,2]
        @test collect(tensor((a,))) == [1,2]

        @inferred tensor( a,b )
        @inferred tensor((a,b))
        @test eltype(tensor( a,b )) == Float64
        @test eltype(tensor((a,b))) == Float64
        @test collect(tensor( a,b )) == [1. 2.; 2. 4.]
        @test collect(tensor((a,b))) == [1. 2.; 2. 4.]
    end

end
