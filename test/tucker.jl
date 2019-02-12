@testset "tucker" begin
    using ApproxTools: tucker
    @testset for TC in rnc((Int,Float64)), TB in rnc((Int,Float64))
        C = rand(TC,2)
        B = (rand(TB,2,2),)
        @test @inferred(tucker(C,B)) ≈ B[1]*C
        @test @inferred(tucker(C,map(B->(C->B*C),B))) ≈ B[1]*C
    end

    @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
        C = rand(TC,2,2)
        B = (rand(TB1,2,2), rand(TB2,2,2))
        @test @inferred(tucker(C,B)) ≈ B[1]*C*transpose(B[2])
        @test @inferred(tucker(C,map(B->(C->B*C),B))) ≈ B[1]*C*transpose(B[2])
    end
end
