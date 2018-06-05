struct MockBasis{M} <: ApproxTools.Basis
    data::M
end
(m::MockBasis)(x) = @view m.data[x,:]
Base.length(m::MockBasis) = size(m.data,2)
Base.eltype(::Type{MockBasis{M}},::Type{<:Integer}) where {M} = eltype(M)
ApproxTools.interpolationpoints(m::MockBasis) = 1:length(m)
ApproxTools.interpolationtransform(m::MockBasis) = m.data[1:length(m),:]

@testset "Basis" begin
    M = [
        1 2 3
        4 5 6
        7 8 9
    ]
    b = MockBasis(M)
    @test eltype(b,1) == eltype(typeof(b),Int)
    @test collect(b,2:3) == M[2:3,:]

    p = @inferred(interpolate(identity, b))
    @test coeffs(p) == M*(1:3)
    @test basis(p) == (b,)

    p = @inferred(interpolate(*, (b,b)))
    @test coeffs(p) == M*(1:3)*(M*(1:3))'
    @test basis(p) == (b,b)
end

@testset "LinearCombination" begin
    using ApproxTools: LinearCombination

    @testset for TC in rnc((Int,Float64)), TB in rnc((Int,Float64))
        C = rand(TC,2)
        B = rand(TB,2,2)

        p = @inferred(LinearCombination(C, MockBasis(B)))
        @test @inferred(p( 1  )) ≈ RowVector(B[1,:])*C
        @test @inferred(p((1,))) ≈ RowVector(B[1,:])*C
        @test @inferred(p( 2  )) ≈ RowVector(B[2,:])*C

        @test_throws MethodError p( [1,2]  )
        @test_throws MethodError p(([1,2],))
    end

    @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
        C = rand(TC,2,2)
        B = (rand(TB1,2,2), rand(TB2,2,2))

        p = @inferred(LinearCombination(C, MockBasis.(B)))
        @test @inferred(p( 1,1 )) ≈ RowVector(B[1][1,:])*C*B[2][1,:]
        @test @inferred(p((1,1))) ≈ RowVector(B[1][1,:])*C*B[2][1,:]
        @test @inferred(p( 1,1:2 )) ≈ RowVector(B[1][1,:])*C*transpose(B[2][1:2,:])
        @test @inferred(p((1,1:2))) ≈ RowVector(B[1][1,:])*C*transpose(B[2][1:2,:])
        @test @inferred(p( 1:2,1:2 )) ≈ B[1][1:2,:]*C*transpose(B[2][1:2,:])
        @test @inferred(p((1:2,1:2))) ≈ B[1][1:2,:]*C*transpose(B[2][1:2,:])
    end
end
