mutable struct MockCartesianFunction
    called_grideval::Bool
end
function ApproxTools.grideval(f::MockCartesianFunction, x::NTuple{N,Any}) where {N}
    f.called_grideval = true
    return broadcast(*,ApproxTools.reshape4grideval(x)...)
end

@testset "grideval" begin
    a = [1,2]
    b = [1.,2.]

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
    @inferred grideval((x,y)->x*y,  b )
    @inferred grideval(*,  1,b )
    @inferred grideval(*,  a,b )
    @inferred grideval(*, (a,b))
    @test typeof(grideval(*,  1,1.0)) == Float64
    @test eltype(grideval((x,y)->x*y, b)) == Float64
    @test eltype(grideval(*,  1,b )) == Float64
    @test eltype(grideval(*,  a,b )) == Float64
    @test eltype(grideval(*, (a,b))) == Float64
    @test grideval(*,  1,1.0 ) == 1.0
    @test collect(grideval(*,  1,b )) == [1. 2.]
    @test collect(grideval((x,y)->x*y,  b )) == [1. 2.; 2. 4.]
    @test collect(grideval(*,  a,b )) == [1. 2.; 2. 4.]
    @test collect(grideval(*, (a,b))) == [1. 2.; 2. 4.]

    m = MockCartesianFunction(false)
    f = let m = m; @gridfun( x -> x + $m(x) ); end
    @test @inferred(grideval(f, a)) ≈ 2*a
    @test m.called_grideval

    m = MockCartesianFunction(false)
    f = let m = m; @gridfun( (x,y) -> x + $m(x,y) ); end
    @test @inferred(grideval(f, a,b)) ≈ a .+ a .*b'
    @test m.called_grideval
end
