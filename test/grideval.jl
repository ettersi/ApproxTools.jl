struct MockCartesianFunction end
ApproxTools.grideval(::MockCartesianFunction, x::NTuple{N,Any}) where {N} = x

@testset "grideval" begin
    a = [1,2]
    b = [1.,2.]

    @test ApproxTools.gridevalshape((a,b),1) == (2,1)
    @test ApproxTools.gridevalshape((a,b),2) == (1,2)
    @test ApproxTools.gridevalshape((a,1),1) == (2,1)
    @test ApproxTools.reshape4grideval(a,(a,b),1) == reshape(a,(2,1))
    @test ApproxTools.reshape4grideval(b,(a,b),2) == reshape(b,(1,2))
    @test ApproxTools.reshape4grideval(a,(a,1),1) == reshape(a,(2,1))
    @test ApproxTools.reshape4grideval(1,(a,1),2) == 1

    @inferred grideval(*,  1 )
    @inferred grideval(*,  a )
    @inferred grideval(*, (a,))
    @test typeof(grideval(*,  1  )) == Int
    @test eltype(grideval(*,  a  )) == Int
    @test eltype(grideval(*, (a,))) == Int
    @test grideval(*,  1  ) == 1
    @test collect(grideval(*,  a  )) == [1,2]
    @test collect(grideval(*, (a,))) == [1,2]

    @inferred grideval(*,  1,1.0 )
    @inferred grideval(*,  1,b )
    @inferred grideval(*,  a,b )
    @inferred grideval(*, (a,b))
    @test typeof(grideval(*,  1,1.0)) == Float64
    @test eltype(grideval(*,  1,b )) == Float64
    @test eltype(grideval(*,  a,b )) == Float64
    @test eltype(grideval(*, (a,b))) == Float64
    @test grideval(*,  1,1.0 ) == 1.0
    @test collect(grideval(*,  1,b )) == [1. 2.]
    @test collect(grideval(*,  a,b )) == [1. 2.; 2. 4.]
    @test collect(grideval(*, (a,b))) == [1. 2.; 2. 4.]

    @test grideval(MockCartesianFunction(), a,b) == (a,b)
end
