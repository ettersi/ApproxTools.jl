@testset "bernstein" begin

    @testset "rsmo" begin
        for T in rnc(Reals)
            @inferred rsmo(one(T))
        end

        @test rsmo(π) ≈ rsmo(big(π))
        @test rsmo(π*im) ≈ rsmo(big(π)*im)

        # Because integer types do not have a signed zero, (-im)^2 gets mapped
        # to the second quadrant (-1+0im) instead of the third (-1-im).
        # Make sure rsmo takes care of this case.
        @test rsmo(-im) == rsmo(-1.0im)

        x = -2.0; @test rsmo(complex(x,0.0)) == rsmo(complex(x,-0.0))
        x = -1.0; @test rsmo(complex(x,0.0)) == rsmo(complex(x,-0.0))
        x =  1.0; @test rsmo(complex(x,0.0)) == rsmo(complex(x,-0.0))
        x =  2.0; @test rsmo(complex(x,0.0)) == rsmo(complex(x,-0.0))

        y = -2.0; @test rsmo(complex(0.0,y)) == rsmo(complex(-0.0,y))
        y = -1.0; @test rsmo(complex(0.0,y)) == rsmo(complex(-0.0,y))
        y =  0.0; @test rsmo(complex(0.0,y)) == rsmo(complex(-0.0,y))
        y =  1.0; @test rsmo(complex(0.0,y)) == rsmo(complex(-0.0,y))
        y =  2.0; @test rsmo(complex(0.0,y)) == rsmo(complex(-0.0,y))

        x = -0.5; @test rsmo(complex(x,0.0)) == -rsmo(complex(x,-0.0))
        x =  0.0; @test rsmo(complex(x,0.0)) == -rsmo(complex(x,-0.0))
        x =  0.5; @test rsmo(complex(x,0.0)) == -rsmo(complex(x,-0.0))
    end

    @testset "jouk" begin
        for T in rnc(Reals)
            @inferred jouk(one(T))
            @inferred ijouk(one(T))
            @inferred ijouk(one(T),Val(true))
            @inferred ijouk(one(T),Val(false))
            @inferred ijoukt(one(T))
            @inferred ijoukt(one(T),Val(true))
            @inferred ijoukt(one(T),Val(false))
        end

        @test jouk(ijouk(π   )) ≈ π
        @test jouk(ijouk(π*im)) ≈ π*im
        @test jouk(ijouk(big(π)   )) ≈ π
        @test jouk(ijouk(big(π)*im)) ≈ π*im
        @test jouk(ijoukt(1/π )) ≈ 1/π
        @test jouk(ijoukt(π*im)) ≈ π*im
        @test jouk(ijoukt(1/big(π) )) ≈ 1/big(π)
        @test jouk(ijoukt(big(π)*im)) ≈ π*im
    end

    @testset "semi axis" begin
        for T in rnc(Reals)
            @inferred semimajor(one(T))
            @inferred semiminor(one(T))
            @inferred radius(one(T))
            @inferred radiust(one(T))
        end

        @test semimajor(π) ≈ π
        @test semimajor(big(π)) ≈ big(π)
        @test semiminor(π*im) ≈ π
        @test semiminor(big(π)*im) ≈ big(π)
        @test radius(1) ≈ 1
        @test radius(1+1/4) ≈ 2
        @test radius(1+1/big(4)) ≈ big(2)
        @test radiust(1) ≈ 1
        @test radiust(complex(1+1/4,0.0)) ≈ 2
        @test radiust(complex(1+1/big(4),big(0))) ≈ big(2)
        @test radiust(complex(1+1/4,-0.0)) ≈ 0.5
    end

end
