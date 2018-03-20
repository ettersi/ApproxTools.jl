testvalues(T::Type{<:Real}) = T[0,1]
testvalues(T::Type{<:Complex}) = T[0,1+im]

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
                    x  in testvalues.(rnc(T)),
                    y  in ((), (T->(testvalues(T).+2,)).(rnc(T))...),
                    y2 in ((), (T->(testvalues(T).+2,)).(rnc(T))...)
                n = length(x)
                T = float(promote_type(eltype.((x,y...,y2...))...))
                s,w = @inferred(baryweights(x,y...,y2...))
                @test typeof(s) == eltype(w) == T
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
        @testset "1D" begin
            @testset for T in Reals
                @testset for
                        x in testvalues.(rnc(T)),
                        f in testvalues.(rnc(T)),
                        xx in testvalues.(rnc(T))
                    sw = baryweights(x)
                    @test eltype(@inferred(bary(x,sw,f,xx[1]))) == float(promote_type(eltype.((x,f,xx))...))
                    @test bary(x,sw,f,xx[1]) == xx[1]
                    @test bary(x,sw,f,sum(xx)) ≈ (f[1]*(x[2]-sum(xx)) + f[2]*(sum(xx)-x[1]))/(x[2]-x[1])
                end
            end
        end

        @testset for T in Reals
            @testset for
                    x1 in testvalues.(rnc(T)),
                    x2 in testvalues.(rnc(T)),
                    f in testvalues.(rnc(T)),
                    xx1 in testvalues.(rnc(T)),
                    xx2 in testvalues.(rnc(T))
                x = (x1,x2)
                f = f.*f'
                xx = (xx->[xx[1],2*xx[2]]).((xx1,xx2))
                sw = baryweights.(x)
                @test eltype(@inferred(bary(x,sw,f,(xx[1][1],xx[2][1])))) == float(promote_type(eltype.((x...,f,xx...))...))
                @test first(bary(x,sw,f,(xx[1][1],xx[2][1]))) == f[1,1]
                @test bary(x,sw,f,xx) ≈ [
                    (
                        f[1,1]*(x[1][2]-xx1)*(x[2][2]-xx2) +
                        f[2,1]*(xx1-x[1][1])*(x[2][2]-xx2) +
                        f[1,2]*(x[1][2]-xx1)*(xx2-x[2][1]) +
                        f[2,2]*(xx1-x[1][1])*(xx2-x[2][1])
                    ) / ((x[1][2]-x[1][1])*(x[2][2]-x[2][1]))
                    for xx1 in xx[1], xx2 in xx[2]
                ]
            end
        end
    end

    @testset "interpolate" begin
        p = @inferred(interpolate([0,1],[0,1]))
        @test @inferred(p(0)) == 0
        @test @inferred(p(0.5)) ≈ 0.5
    end

end
