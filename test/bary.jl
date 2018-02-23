@testset "barycentric interpolation" begin

@testset "geometric_mean_distance" for T in Floats
    @inferred geometric_mean_distance(zeros(T,3))

    x = T[γ]; @test geometric_mean_distance(x) ≈ 1
    x = T[γ,e]; @test geometric_mean_distance(x) ≈ abs(x[1]-x[2])
    x = T[γ,e,π]; @test geometric_mean_distance(x) ≈ abs((x[1]-x[2])*(x[1]-x[3])*(x[2]-x[3]))^(T(1)/3)

    n = 1000
    x = cos.(T(π)/(n-1)*collect(0:n-1))
    @test geometric_mean_distance(x) ≈ 0.5 atol=0.01
end


@testset "baryweights" for T in Floats
    @inferred baryweights(zeros(T,3))

    n = 5
    x = cos.(T(π)/(n-1)*collect(0:n-1))
    w = baryweights(x)
    @test 2w[1] ≈ - w[2]
    @test  w[2] ≈ - w[3]
    @test  w[3] ≈ - w[4]
    @test  w[4] ≈ -2w[5]

    n = 1000
    x = cos.(T(π)/(n-1)*collect(0:n-1))
    w = baryweights(x)
    @test all(isfinite.(w))
    @test !any(iszero.(w))
end

reptuple(v,n) = ntuple(i->v,n)

@testset "bary" for T in Floats
    @testset "type stability" begin
        @inferred bary(zeros(T,3),zeros(T,3),zeros(T,3),zero(T))
        @inferred bary((zeros(T,3),),(zeros(T,3),),zeros(T,3),(zero(T),))
        @inferred bary((zeros(T,3),),(zeros(T,3),),zeros(T,3),(zeros(T,2),))
    end

    x = T[-1,0,1]
    w = baryweights(x)
    f = T[1,0,1]

    @testset "pointwise" begin
        @test bary(x,w,f,T(1)) == T(1)
        @test bary(x,w,f,T(π)) ≈  T(π)^2
    end

    @testset "cartesian 1D" begin
        @test bary((x,),(w,),f,(T(1),))[1] == T(1)
        @test bary((x,),(w,),f,(T(π),))[1] ≈  T(π)^2

        @test bary((x,),(w,),f,(T[1],))[1] == T(1)
        @test bary((x,),(w,),f,(T[π],))[1] ≈  T(π)^2

        @test bary((x,),(w,),f,([T(1),T(π)],)) ≈ [T(1),T(π)^2]
    end

    @testset "cartesian 2D" begin
        x = (T[-1,0,1],T[0,1])
        w = baryweights.(x)
        f = T[0 1; 0 0; 0 1]
        @test bary(x,w,f,(T[1,π],T[0,e])) ≈ [zero(T) T(e); zero(T) T(π)^2*T(e)]
    end

    @testset "cartesian 3D" begin
        x = (T[-1,0,1],T[0,1],T[-1,1])
        w = baryweights.(x)
        f = zeros(T,3,2,2)
        f[1,2,2] = 1
        f[3,2,2] = 1
        b = zeros(T,2,2,2)
        b[1,2,2] =        T(e)*(T(γ)+1)/2
        b[2,2,2] = T(π)^2*T(e)*(T(γ)+1)/2
        @test bary(x,w,f,(T[1,π],T[0,e],T[-1,γ])) ≈ b
    end
end

end
