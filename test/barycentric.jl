@testset "barycentric" begin

    @testset "LogNumber" begin
        using ApproxTools: lognumber, LogNumber, logabs

        @test lognumber(Float64) == LogNumber{Float64,Float64}
        @test lognumber(ComplexF64) == LogNumber{ComplexF64,Float64}
        @test lognumber(1.0) == LogNumber(1.0,0.0)

        @test sign(-exp(2.0)) == -1.0
        @test logabs(-exp(2.0)) == 2.0
        @test sign(LogNumber(-1.0,2.0)) == -1.0
        @test logabs(LogNumber(-1.0,2.0)) == 2.0

        @test one(lognumber(Float64)) == 1.0
        @test one(lognumber(ComplexF64)) == 1.0

        @test convert(LogNumber, LogNumber(1.0,0.0)) == LogNumber(1.0,0.0)
        @test convert(LogNumber{Float64,Float64}, LogNumber(1.0,0.0)) == LogNumber(1.0,0.0)
        @test convert(LogNumber{ComplexF64,Float64}, LogNumber(1.0,0.0)) == LogNumber(1.0+0im,0.0)
        @test convert(LogNumber, 1.0) == LogNumber(1.0,0.0)
        @test convert(LogNumber{ComplexF64,Float64}, 1.0) == LogNumber(1.0+0im,0.0)
        @test convert(Float64, LogNumber(1.0,0.0)) == 1.0

        @test LogNumber(-1.0,2.0) * exp(2.0) == - exp(4.0)
        @test LogNumber(-1.0,2.0) / exp(2.0) == - exp(0.0)
        @test LogNumber(  im,2.0) * exp(2.0) == im * exp(4.0)
        @test LogNumber(  im,2.0) / exp(2.0) == im * exp(0.0)
        @test exp(2.0) * LogNumber(-1.0,2.0) == - exp(4.0)
        @test exp(2.0) / LogNumber(-1.0,2.0) == - exp(0.0)
        @test exp(2.0) * LogNumber(  im,2.0) ==   im * exp(4.0)
        @test exp(2.0) / LogNumber(  im,2.0) == - im * exp(0.0)
        @test LogNumber(1.0, 2.0) * LogNumber(-1.0,2.0) == - exp(4.0)
        @test LogNumber(1.0, 2.0) / LogNumber(-1.0,2.0) == - exp(0.0)
        @test LogNumber(1.0, 2.0) * LogNumber(  im,2.0) ==   im * exp(4.0)
        @test LogNumber(1.0, 2.0) / LogNumber(  im,2.0) == - im * exp(0.0)
    end

    @testset "prodpot" begin
        using ApproxTools: prodpot, lognumber, logabs

        testx̂(::Type{T}) where {T <: Real} = T(1)
        testx̂(::Type{T}) where {T <: Complex} = testx̂(real(T))+im
        testx(::Type{T}) where {T <: Real} = T[2,3,4]
        testx(::Type{T}) where {T <: Complex} = testx(real(T)).+im

        @testset for T1 = rnc(Reals), T2 = rnc(Reals)
            x̂ = testx̂(T1)
            x = testx(T2)
            @test float(@inferred(prodpot(x̂,x))) ≈ prod(x̂.-x)
        end
        @testset for T = rnc(Reals)
            x = testx(T)
            @test float.(@inferred(prodpot(x))) ≈ [prod(x[1].-x[[2,3]]), prod(x[2].-x[[1,3]]), prod(x[3].-x[[1,2]])]
        end

        @testset "overflow" begin
            x = lognumber(realmax())
            @test logabs(x*x) ≈ 2*log(realmax())
        end
    end

    @testset "Barycentric" begin
        using ApproxTools: interpolationpoints

        x = [1,2,3]
        xx = 4

        b = @inferred(Barycentric(x))
        @test length(b) == 3
        @test eltype(b,1) == Float64
        @test eltype(b,1im) == ComplexF64
        @test interpolationpoints(b) == x
        bv = @inferred(b(xx))
        @test length(bv) == 3
        @test eltype(bv) == Float64

        b = @inferred(Barycentric(x, x->im*sin(x)))
        bv = b(xx)
        @test length(bv) == 3
        @test eltype(bv) == ComplexF64

        b = Barycentric(x)
        bv = @inferred(b(xx))
        s = @inferred(start(bv))
        bxi,s = @inferred(next(bv,s)); @test bxi ≈ (xx-x[2])*(xx-x[3])/((x[1]-x[2])*(x[1]-x[3])); @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi ≈ (xx-x[1])*(xx-x[3])/((x[2]-x[1])*(x[2]-x[3])); @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi ≈ (xx-x[1])*(xx-x[2])/((x[3]-x[1])*(x[3]-x[2])); @test  done(bv,s)

        bv = @inferred(b(x[2]))
        s = @inferred(start(bv))
        bxi,s = @inferred(next(bv,s)); @test bxi == 0; @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi == 1; @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi == 0; @test  done(bv,s)

        b = Barycentric(x, x->im*sin(x))
        bv = @inferred(b(xx))
        s = @inferred(start(bv))
        bxi,s = @inferred(next(bv,s)); @test bxi ≈ sin(xx)/sin(x[1])*(xx-x[2])*(xx-x[3])/((x[1]-x[2])*(x[1]-x[3])); @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi ≈ sin(xx)/sin(x[2])*(xx-x[1])*(xx-x[3])/((x[2]-x[1])*(x[2]-x[3])); @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi ≈ sin(xx)/sin(x[3])*(xx-x[1])*(xx-x[2])/((x[3]-x[1])*(x[3]-x[2])); @test  done(bv,s)

        bv = @inferred(b(x[2]))
        s = @inferred(start(bv))
        bxi,s = @inferred(next(bv,s)); @test bxi == 0; @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi == 1; @test !done(bv,s)
        bxi,s = @inferred(next(bv,s)); @test bxi == 0; @test  done(bv,s)
    end

    @testset "Barycentric interpolation" begin
        @testset "1D" begin
            f = x->x^2
            b = @inferred(Barycentric([-1,0,1]))
            p = @inferred(interpolate(f,b))
            x̂ = linspace(-3,3,10)
            @test p.(x̂) ≈ f.(x̂)
        end

        @testset "2D" begin
            f = *
            b = @inferred(Barycentric([-1,0,1]))
            p = @inferred(interpolate(f,(b,b)))
            x̂ = linspace(-3,3,10)
            @test p(x̂,x̂) ≈ f.(x̂,x̂')
        end
    end
end
