@testset "bernstein" begin

    @testset "rsmo" begin
        for T in rnc(Reals)
            @inferred rsmo(one(T))
        end

        @test rsmo(π) ≈ rsmo(big(π))
        @test rsmo(π*im) ≈ rsmo(big(π)*im)

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
        end

        @test jouk(ijouk(π   )) ≈ π
        @test jouk(ijouk(π*im)) ≈ π*im
        @test jouk(ijouk(big(π)   )) ≈ π
        @test jouk(ijouk(big(π)*im)) ≈ π*im
    end

    @testset "semi axis" begin
        for T in rnc(Reals)
            @inferred semimajor(one(T))
            @inferred semiminor(one(T))
        end

        @test semimajor(π) ≈ π
        @test semimajor(big(π)) ≈ big(π)
        @test semiminor(π*im) ≈ π
        @test semiminor(big(π)*im) ≈ big(π)
    end

end
