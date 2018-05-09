struct MockBasis{M} <: ApproxTools.Basis
    data::M
end
(m::MockBasis)(x) = @view m.data[:,x]
Base.length(m::MockBasis) = size(m.data,1)
Base.eltype(::Type{MockBasis{M}},::Type{<:Integer}) where {M} = eltype(M)

@testset "LinearCombination" begin
    using ApproxTools: LinearCombination

    @testset for TC in rnc((Int,Float64)), TB in rnc((Int,Float64))
        C = TC[1,2]
        B = (TB[3 4; 5 6],)

        p = @inferred(LinearCombination(C, MockBasis.(B)))
        @test @inferred(p( 1  )) == (B[1][:,1]'*C)[1]
        @test @inferred(p((1,))) == (B[1][:,1]'*C)[1]
        @test @inferred(p( 2  )) == (B[1][:,2]'*C)[1]

        @test_throws MethodError p( [1,2]  )
        @test_throws MethodError p(([1,2],))
    end

    @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
        C = TC[1 2; 3 4]
        B = (TB1[5 6; 7 8], TB2[9 10; 11 12])

        p = @inferred(LinearCombination(C, MockBasis.(B)))
        @test @inferred(p( 1,1 )) == B[1][:,1]'*C*B[2][:,1]
        @test @inferred(p((1,1))) == B[1][:,1]'*C*B[2][:,1]
        @test @inferred(p( 1,1:2 )) == B[1][:,1]'*C*B[2][:,1:2]
        @test @inferred(p((1,1:2))) == B[1][:,1]'*C*B[2][:,1:2]
        @test @inferred(p( 1:2,1:2 )) == B[1][:,1:2]'*C*B[2][:,1:2]
        @test @inferred(p((1:2,1:2))) == B[1][:,1:2]'*C*B[2][:,1:2]
    end
end
