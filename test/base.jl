struct MockInterpolation <: ApproxTools.InterpolationAlgorithm end

ApproxTools.interpolate(
    x::NTuple{N,<:AbstractVector},
    f::AbstractArray{<:Any,N},
    y::NTuple{N,<:AbstractVector},
    y2::NTuple{N,<:AbstractVector},
    ::MockInterpolation
) where {N} = x,f,y,y2

@testset "base" begin

@testset "interpolate" begin

    @testset "1D" begin
        x = [0]
        f = [1]
        y = [2]
        y2 = [3]
        e = EmptyVector()

        # Test arguments are passed correctly
        @test interpolate(x,f,MockInterpolation()) == ((x,),f,(e,),(e,))
        @test interpolate(x,f,y,MockInterpolation()) == ((x,),f,(y,),(e,))
        @test interpolate(x,f,y,y2,MockInterpolation()) == ((x,),f,(y,),(y2,))

        # Make sure default algorithm works
        interpolate(x,f)
        interpolate(x,f,y)
        interpolate(x,f,y,y2)
    end

    @testset "2D definition" begin
        x = ([0],[0])
        f = reshape([1],(1,1))
        y = ([2],[2])
        y2 = ([3],[3])
        e = (EmptyVector(),EmptyVector())

        # Test arguments are passed correctly
        @test interpolate(x,f,MockInterpolation()) == (x,f,e,e)
        @test interpolate(x,f,y,MockInterpolation()) == (x,f,y,e)
        @test interpolate(x,f,y,y2,MockInterpolation()) == (x,f,y,y2)

        # Make sure default algorithm works
        interpolate(x,f)
        interpolate(x,f,y)
        interpolate(x,f,y,y2)
    end

end


end
