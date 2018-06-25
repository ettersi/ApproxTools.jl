struct MockBasis{M} <: ApproxTools.Basis
    data::M
end
(m::MockBasis)(x) = @view m.data[x,:]
Base.length(m::MockBasis) = size(m.data,2)
ApproxTools.interpolationpoints(m::MockBasis) = 1:length(m)
ApproxTools.interpolationtransform(m::MockBasis) = f->m.data*f

struct MockValues{M} <: ApproxTools.BasisValues
    basis::MockBasis{M}
    evaluationpoint::Int
end
Base.eltype(::Type{MockValues{M}}) where {M} = eltype(M)
Base.getindex(m::MockValues,i) = m.basis.data[m.evaluationpoint,i]

@testset "Basis" begin
    M = [
        1 2 3
        4 5 6
        7 8 9
    ]
    b = MockBasis(M)
    @test collect(b,2:3) == M[2:3,:]

    p = @inferred(interpolate(identity, b))
    @test coeffs(p) == M*(1:3)
    @test basis(p) == (b,)

    p = @inferred(interpolate(*, (b,b)))
    @test coeffs(p) == M*(1:3)*(M*(1:3))'
    @test basis(p) == (b,b)

    bv = MockValues(b,1)
    @test length(bv) == length(b)
    @test eltype(bv) == Int
    @test collect(bv) == M[1,:]
end

@testset "LinearCombination" begin
    @testset "scalar 1d" begin
        @testset for TC in rnc((Int,Float64)), TB in rnc((Int,Float64))
            C = rand(TC,2)
            B = rand(TB,2,2)

            p = @inferred(LinearCombination(C, MockBasis(B)))
            @test ApproxTools.GridevalStyle(p) == ApproxTools.GridevalCartesian()
            @test ndims(p) == ndims(typeof(p)) == 1
            @test @inferred(p( 1  )) ≈ RowVector(B[1,:])*C
            @test @inferred(p((1,))) ≈ RowVector(B[1,:])*C
            @test @inferred(p( 2  )) ≈ RowVector(B[2,:])*C

            @test @inferred(p( [1,2]  )) ≈ B[1:2,:]*C
            @test @inferred(p(([1,2],))) ≈ B[1:2,:]*C
        end
    end

    @testset "scalar 2d" begin
        @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
            C = rand(TC,2,2)
            B = (rand(TB1,2,2), rand(TB2,2,2))

            p = @inferred(LinearCombination(C, MockBasis.(B)))
            @test ndims(p) == ndims(typeof(p)) == 2
            @test @inferred(p( 1,1 )) ≈ RowVector(B[1][1,:])*C*B[2][1,:]
            @test @inferred(p((1,1))) ≈ RowVector(B[1][1,:])*C*B[2][1,:]
            @test @inferred(p( 1,1:2 )) ≈ RowVector(B[1][1,:])*C*transpose(B[2][1:2,:])
            @test @inferred(p((1,1:2))) ≈ RowVector(B[1][1,:])*C*transpose(B[2][1:2,:])
            @test @inferred(p( 1:2,1:2 )) ≈ B[1][1:2,:]*C*transpose(B[2][1:2,:])
            @test @inferred(p((1:2,1:2))) ≈ B[1][1:2,:]*C*transpose(B[2][1:2,:])
        end
    end

    @testset "mat/matvec 1d" begin
        C = rand(5)
        p = LinearCombination(C,Monomial(length(C)))
        x = linspace(-1,1,11)
        @test diag(@inferred(p(Diagonal(x)))) == p.(x)
        @test @inferred(p(Diagonal(x),ones(x))) == p.(x)
    end


end

@testset "Semiseparated" begin
    @testset for TC in rnc((Int,Float64)), TB1 in rnc((Int,Float64)), TB2 in rnc((Int,Float64))
        C = rand(TC,2,2)
        B = (rand(TB1,2,2), rand(TB2,2,2))

        p = @inferred(Semiseparated(*,(x->x+1,x->x-1)))
        @test ApproxTools.GridevalStyle(p) == ApproxTools.GridevalCartesian()
        @test ndims(p) == ndims(typeof(p)) == 2
        @test @inferred(p( 1,1 )) ≈ 0
        @test @inferred(p((1,1))) ≈ 0
        @test @inferred(p( 1,1:2 )) ≈ [0 4]
        @test @inferred(p((1,1:2))) ≈ [0 4]
        @test @inferred(p( 1:2,1:2 )) ≈ [0 4; 0 12]
        @test @inferred(p((1:2,1:2))) ≈ [0 4; 0 12]
    end
end

@testset "Monomial" begin
    b = Monomial(5)
    x = linspace(-1,1,11)
    monos = [one.(x), x, @.(x^2), @.(x^3), @.(x^4)]
    v = rand(length(x))
    @test collect(b,x) ≈ hcat(monos...)
    @test all(collect(b(full(Diagonal(x)))) .≈ Diagonal.(monos))
    @test all(collect(b(Diagonal(x))) .≈ Diagonal.(monos))
    @test all(collect(b(Diagonal(x),v)) .≈ Diagonal.(monos).*(v,))

    for bv = (@inferred(b(0)), @inferred(b(zeros(2,2))), @inferred(b(zeros(2,2),zeros(2))))
        bvs = @inferred start(bv)
        _,bvs = @inferred next(bv,bvs)
        @inferred done(bv,bvs)
    end
end

@testset "Chebyshev" begin
    using ApproxTools: interpolationpoints, interpolationtransform

    b = Chebyshev(5)
    x = linspace(-1,1,11)
    chebpolys = [one.(x), x, @.(2x.^2-1), @.(4x^3-3x), @.(8x^4-8x^2+1)]
    v = rand(length(x))
    @test collect(b,x) ≈ hcat(chebpolys...)
    @test all(collect(b(full(Diagonal(x)))) .≈ Diagonal.(chebpolys))
    @test all(collect(b(Diagonal(x))) .≈ Diagonal.(chebpolys))
    @test all(collect(b(Diagonal(x),v)) .≈ Diagonal.(chebpolys).*(v,))

    for bv = (@inferred(b(0)), @inferred(b(zeros(2,2))), @inferred(b(zeros(2,2),zeros(2))))
        bvs = @inferred start(bv)
        _,bvs = @inferred next(bv,bvs)
        @inferred done(bv,bvs)
    end

    @testset for n = 0:5, T = rnc((Int,Float32,Float64))
        b = @inferred(Chebyshev(n))
        a = rand(T,n)
        @test @inferred(collect(b,@inferred(interpolationpoints(b))))*@inferred(interpolationtransform(b)(a)) ≈ a
        A = rand(T,n,n)
        @test @inferred(collect(b,@inferred(interpolationpoints(b))))*@inferred(interpolationtransform(b)(A)) ≈ A
    end
end

@testset "Weighted" begin
    using ApproxTools: interpolationpoints, interpolationtransform

    b = Weighted(Chebyshev(5), x->exp(-x^2))
    x = linspace(-1,1,11)
    @test collect(b,x) ≈ exp.(-x.^2).*hcat(one.(x), x, @.(2x.^2-1), @.(4x^3-3x), @.(8x^4-8x^2+1))

    @testset for n = 0:5, T = rnc((Int,Float32,Float64))
        b = @inferred(Weighted(Chebyshev(n), x->exp(-x^2)))
        a = rand(T,n)
        @test @inferred(collect(b,@inferred(interpolationpoints(b))))*@inferred(interpolationtransform(b)(a)) ≈ a
        A = rand(T,n,n)
        @test @inferred(collect(b,@inferred(interpolationpoints(b))))*@inferred(interpolationtransform(b)(A)) ≈ A
    end
end
