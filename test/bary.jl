@testset "barycentric interpolation" begin

@testset "baryweights" for T in (Floats...,complex.(Floats)...)
    @testset "polynomial" begin
        @inferred baryweights(zeros(T,3))

        n = 5
        x = myrand(T,n)
        w = baryweights(x)
        c = w[1] * prod(x[1].-x[2:n])
        for i = 2:length(x)
            @test w[i] * prod(x[i].-x[[1:i-1;i+1:n]]) ≈ c
        end

        n = 1000
        x = myrand(T)*cos.(T(π)/(n-1)*collect(0:n-1))
        w = baryweights(x)
        @test all(isfinite.(w))
        @test !any(iszero.(w))
        @test 2w[1] ≈ -w[2]         rtol=sqrt(n*eps(real(T)))
        @test -w[n-1] ≈ 2w[n]       rtol=sqrt(n*eps(real(T)))
        @test w[2:n-2] ≈ -w[3:n-1]  rtol=sqrt(n*eps(real(T)))
    end

    @testset "rational" begin
        @inferred baryweights(zeros(T,3), zeros(T,3))

        n = 5
        x = myrand(T,n)
        y = myrand(T,3)
        w = baryweights(x,y)
        c = w[1] * prod(x[1].-x[2:n]) / prod(x[1].-y)
        for i = 2:length(x)
            @test w[i] * prod(x[i].-x[[1:i-1;i+1:n]]) / prod(x[i].-y) ≈ c
        end
    end

    @testset "rational csym" begin
        @inferred baryweights(zeros(T,3), (), zeros(T,3))

        n = 5
        x = myrand(T,n)
        y2 = myrand(T,3)
        w = baryweights(x,(),y2)
        c = w[1] * prod(x[1].-x[2:n]) / prod(x[1]^2.- y2)
        for i = 2:length(x)
            @test w[i] * prod(x[i].-x[[1:i-1;i+1:n]]) / prod(x[i].^2.- y2) ≈ c
        end
    end

    @testset "rational combined" begin
        @inferred baryweights(zeros(T,3), zeros(T,3), zeros(T,3))

        n = 5
        x = myrand(T,n)
        y = myrand(T,3)
        y2 = myrand(T,2)
        w = baryweights(x,y,y2)
        c = w[1] * prod(x[1].-x[2:n]) / (prod(x[1].-y)*prod(x[1]^2.- y2))
        for i = 2:length(x)
            @test w[i] * prod(x[i].-x[[1:i-1;i+1:n]]) / (prod(x[i].-y)*prod(x[i].^2.- y2)) ≈ c
        end
    end
end

reptuple(v,n) = ntuple(i->v,n)

@testset "bary" for T in (Floats...,complex.(Floats)...)
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
