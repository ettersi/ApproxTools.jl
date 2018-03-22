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

                function mergepoles(W,y,y2)
                    y2h = @. sqrt(complex(-y2))
                    return complex(W)[y..., (.-y2h)..., y2h...]
                end

                function refvalue(x,f,xx,y)
                    @assert length(x) == length(f) == 2
                    return (
                            f[1]*(x[2]-xx)*prod(x[1].-y) +
                            f[2]*(xx-x[1])*prod(x[2].-y)
                        ) / ( prod(xx.-y)*(x[2]-x[1]) )
                end

                @testset "1D" begin
                    @test eltype(@inferred(bary(x,sw,f,xx[1],y,y2))) == W
                    @test bary(x,sw,f,xx[1],y,y2) == refvalue(x,f,xx[1],mergepoles(W,y,y2))
                    @test bary(x,sw,f,xx[2],y,y2) ≈ refvalue(x,f,xx[2],mergepoles(W,y,y2))
                end

                @testset "2D" begin
                    # We actually should also test with different types in each  pair, but doing
                    # so makes the number of test cases explode
                    tx = (x,x.+1)
                    tf = (f,f.+1)
                    txx = (xx,xx.+1)
                    ty = (y,y.+1)
                    ty2 = (y2,y2.+1)
                    tsw = baryweights.(tx,ty,ty2)

                    pref = map((x,f,xx,y) -> [refvalue(x,f,xx,y) for xx in xx], tx,tf,txx, mergepoles.(W,ty,ty2))
                    p11 =  first(@inferred(bary(tx,tsw,tf[1]*transpose(tf[2]),(txx[1][1],txx[2][1]),ty,ty2)))

                    @test typeof(p11) == W
                    @test p11 == tf[1][1]*tf[2][1]
                    @test @inferred(bary(tx,tsw,tf[1]*transpose(tf[2]),txx,ty,ty2)) ≈ pref[1]*transpose(pref[2])
                end
            end
        end
    end

    @testset "interpolate" begin
        @testset "1D" begin
            x = [0]
            f = [1]
            y = [2]
            y2 = [3]

            p = @inferred07(interpolate(x,f))
            @test @inferred(p(x[1])) == f[1]
            @test @inferred(p((x[1],))) == f[1]
            @test @inferred(p(x)) == f
            @test @inferred(p((x,))) == f

            p = @inferred07(interpolate(x,f; poles=y))
            @test @inferred(p(x[1])) == f[1]
            @test @inferred(p((x[1],))) == f[1]
            @test @inferred(p(x)) == f
            @test @inferred(p((x,))) == f
            @test abs(p(y[1])) > 1e15

            p = @inferred07(interpolate(x,f; cspoles=y2))
            @test @inferred(p(x[1])) == f[1]
            @test @inferred(p((x[1],))) == f[1]
            @test @inferred(p(x)) == f
            @test @inferred(p((x,))) == f
            @test abs(p(im*sqrt(y2[1]))) > 1e15

            p = @inferred07(interpolate(x,f; poles=y, cspoles=y2))
            @test @inferred(p(x[1])) == f[1]
            @test @inferred(p((x[1],))) == f[1]
            @test @inferred(p(x)) == f
            @test @inferred(p((x,))) == f
            @test abs(p(y[1])) > 1e15
            @test abs(p(im*sqrt(y2[1]))) > 1e15
        end

        @testset "2D" begin
            x = ([0],[1])
            f = reshape([2],(1,1))
            y = ([3],[4])
            y2 = ([5],[6])

            p = @inferred07(interpolate(x,f))
            @test @inferred(p(x[1][1],x[2][1])) == f[1,1]
            @test @inferred(p((x[1][1],x[2][1]))) == f[1,1]
            @test @inferred(p(x...)) == f
            @test @inferred(p(x)) == f

            p = @inferred07(interpolate(x,f; poles=y))
            @test @inferred(p(x[1][1],x[2][1])) == f[1,1]
            @test @inferred(p((x[1][1],x[2][1]))) == f[1,1]
            @test @inferred(p(x...)) == f
            @test @inferred(p(x)) == f
            @test abs(p((y[1][1],y[2][1]))) > 1e15


            p = @inferred07(interpolate(x,f; cspoles = y2))
            @test @inferred(p(x[1][1],x[2][1])) == f[1,1]
            @test @inferred(p((x[1][1],x[2][1]))) == f[1,1]
            @test @inferred(p(x...)) == f
            @test @inferred(p(x)) == f
            @test abs(p(im.*sqrt.((y2[1][1],y2[2][1])))) > 1e15

            p = @inferred07(interpolate(x,f; poles = y, cspoles = y2))
            @test @inferred(p(x[1][1],x[2][1])) == f[1,1]
            @test @inferred(p((x[1][1],x[2][1]))) == f[1,1]
            @test @inferred(p(x...)) == f
            @test @inferred(p(x)) == f
            @test abs(p((y[1][1],y[2][1]))) > 1e15
            @test abs(p(im.*sqrt.((y2[1][1],y2[2][1])))) > 1e15
        end
    end

end
