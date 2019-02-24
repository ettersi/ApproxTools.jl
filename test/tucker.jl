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

    @testset for TC in rnc(Float64), TB1 in rnc(Float64), TB2 in rnc(Float64), TB3 in rnc(Float64)
        C = rand(TC,2,2,2)
        B = (rand(TB1,2,2), rand(TB2,2,2), rand(TB3,2,2))
        R = reshape(transpose(B[3]*reshape(transpose(B[2]*reshape(transpose(B[1]*reshape(C,(2,4))),(2,4))),(2,4))),(2,2,2))
        @test @inferred(tucker(C,B)) ≈ R
        @test @inferred(tucker(C,map(B->(C->B*C),B))) ≈ R
    end
end
