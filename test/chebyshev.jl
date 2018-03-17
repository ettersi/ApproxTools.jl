@testset "Chebyshev" begin

@testset "chebpoints" for T in (Floats...,complex.(Floats)...)
    @test eltype(collect(chebpoints(T,3))) == T

    @test chebpoints(T,0) ≈ T[]
    @test chebpoints(T,1) ≈ T[0]  atol = eps(real(T))
    @test chebpoints(T,2) ≈ T[-1,1]
    @test chebpoints(T,3) ≈ T[-1,0,1]
    @test chebpoints(T,4) ≈ T[-1,-0.5,0.5,1]
    @test chebpoints(T,5) ≈ T[-1,-1/sqrt(T(2)),0,1/sqrt(T(2)),1]
end

@testset "baryweights" for T in (Floats...,complex.(Floats)...)
    x = chebpoints(T,5)
    p = interpolate(x,x.^2)
    @test p(π) ≈ T(π)^2
end

function unit(T,n,k)
    e = zeros(T,n)
    e[k] = 1
    return e
end

@testset "chebcoeffs" for T in (BitsFloats...,complex.(BitsFloats)...)
    n = 10
    x = chebpoints(T,n)
    p = [ ones(x), x, @.(2x^2-1), @.(4x^3-3x), @.(8x^4-8x^2+1) ]

    @testset "1D" begin
        for i = 1:length(p)
            @test chebcoeffs(p[i]) ≈ unit(T,n,i)
        end
    end

    @testset "2D" begin
        for i = 1:length(p), j = 1:length(p)
            @test chebcoeffs(p[i]*p[j]') ≈ unit(T,n,i)*unit(T,n,j)'
        end
    end
end

end
