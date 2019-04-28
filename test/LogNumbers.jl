@testset "LogNumbers" begin
    using ApproxTools: LogNumber, logabs

    @test convert(LogNumber, 1) == LogNumber(1,0)
    @test convert(LogNumber, LogNumber(1,0)) == LogNumber(1,0)
    @test convert(LogNumber{Int,Int}, 1) == LogNumber(1,0)
    @test convert(LogNumber{Int,Int}, LogNumber(1,0)) == LogNumber(1,0)
    @test float(LogNumber(1,0)) == 1.0

    @test logabs(-exp(2.0)) == 2.0
    @test sign(LogNumber(-1,2)) == -1
    @test logabs(LogNumber(-1,2)) == 2

    @test LogNumber(-1,2) * exp(2) ==    - exp(4)
    @test LogNumber(-1,2) / exp(2) ==    - exp(0)
    @test LogNumber(im,2) * exp(2) == im * exp(4)
    @test LogNumber(im,2) / exp(2) == im * exp(0)
    @test exp(2) * LogNumber(-1,2) ==      - exp(4)
    @test exp(2) / LogNumber(-1,2) ==      - exp(0)
    @test exp(2) * LogNumber(im,2) ==   im * exp(4)
    @test exp(2) / LogNumber(im,2) == - im * exp(0)
    @test LogNumber(1, 2) * LogNumber(-1,2) ==      - exp(4)
    @test LogNumber(1, 2) / LogNumber(-1,2) ==      - exp(0)
    @test LogNumber(1, 2) * LogNumber(im,2) ==   im * exp(4)
    @test LogNumber(1, 2) / LogNumber(im,2) == - im * exp(0)
end
