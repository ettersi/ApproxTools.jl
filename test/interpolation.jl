 testx(T) = ( T[0,1], Complex{T}[0,1+im] )
 testf(T) = ( T[1,2], Complex{T}[1,2+im] )
testxx(T) = ( T[0,2], Complex{T}[0,2+im] )
 testy(T) = ( (), T[3], Complex{T}[im] )
testy2(T) = ( (), T[3], Complex{T}[im] )

@testset "interpolation" begin

    function refbaryweights(x,y=(),y2=())
        T = float(promote_type(eltype.((x,y,y2))...))

        # Make sure we compute everything at highest precision
        x  = T[x...]
        y  = T[y...]
        y2 = T[y2...]

        n = length(x)
        w = Vector{T}(undef,n)
        for i = 1:n
            w[i] = prod(x[i] .- y) * prod(x[i]^2 .+ y2) / prod(x[i] .- x[[1:i-1;i+1:n]])
        end
        return w
    end

    @testset "baryweights" begin
        # Both baryweights and these tests would work for arbitrary
        # combinations of input types. However, compiling all these
        # test cases takes ages, so we only test input types with the
        # same precision.
        @testset for T in Reals
            @testset for
                    x  in  testx(T),
                    y  in  testy(T),
                    y2 in testy2(T)

                n = length(x)
                s,w = @inferred(baryweights(x,y...,y2...))
                @test typeof(s) == eltype(w) == float(promote_type(eltype.((x,y...,y2...))...))
                @test s^n*w ≈ refbaryweights(x,y...,y2...)
            end
        end

        @testset "over- or underflow" begin
            n = 1000
            x = cos.(Float32(π)/(n-1)*collect(0:n-1))
            s,w = baryweights(x)
            @test isfinite(s) && s != 0
            @test all(isfinite.(w)) && !any(w .== 0)
        end
    end

    @testset "bary" begin
        @testset for T in (Int,)
            @testset "X = $(eltype(x)), F = $(eltype(f)), XX = $(eltype(xx)), Y = $(eltype(y)), Y2 = $(eltype(y2))" for
                    x  in  testx(T),
                    f  in  testf(T),
                    xx in testxx(T),
                    y  in  testy(T),
                    y2 in testy2(T)

                sw = baryweights(x,y,y2)
                W = float(promote_type(eltype.((x,f,xx,y,y2))...))
                y2h = @. sqrt(complex(-y2))
                ya = complex(W)[y..., (.-y2h)..., y2h...]

                function refvalue(x,f,xx,y)
                    @assert length(x) == length(f) == 2
                    return (
                            f[1]*(x[2]-xx)*prod(x[1].-y) +
                            f[2]*(xx-x[1])*prod(x[2].-y)
                        ) / ( prod(xx.-y)*(x[2]-x[1]) )
                end

                @testset "1D" begin
                    @test eltype(@inferred(bary(x,sw,f,xx[1],y,y2))) == W
                    @test bary(x,sw,f,xx[1],y,y2) == refvalue(x,f,xx[1],ya)
                    @test bary(x,sw,f,xx[2],y,y2) ≈ refvalue(x,f,xx[2],ya)
                end

                @testset "2D" begin
                    # We actually should also test with different types in each  pair, but doing
                    # so makes the number of test cases explode
                    tx = (x,x)
                    tsw = (sw,sw)
                    tf = f*transpose(f)
                    txx = (xx,xx)
                    ty = (y,y)
                    ty2 = (y2,y2)

                    p = [refvalue(x,f,xx,ya) for xx in xx]

                    @test eltype(@inferred(bary(tx,tsw,tf,(txx[1][1],txx[2][1]),ty,ty2))) == W
                    @test first(bary(tx,tsw,tf,(txx[1][1],txx[2][1]),ty,ty2)) == tf[1,1]
                    @test bary(tx,tsw,tf,txx,ty,ty2) ≈ p*transpose(p)
                end
            end
        end

    end

    @testset "interpolate" begin
        p = @inferred(interpolate([0,1],[0,1]))
        @test @inferred(p(0.5)) ≈ 0.5
        p = @inferred(interpolate([0,1],[0,1]; poles=[0.5]))
        @test @inferred(p(0.5)) ≈ Inf
    end

end
