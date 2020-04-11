using LinearAlgebra

@testset "inner products" begin
    q = [ x->1/sqrt(2), x->x*sqrt(3/2), x->(3x^2-1)*sqrt(5/8), x->(5x^3-3x)*sqrt(7/8) ]
    p = L2(length(q))
    x = evalpoints(p)
    B = Basis(q)
    Q = [q(x) for x in x, q in q]

    for i = 1:length(q)
        @test @inferred(norm(q[i]    ,p)) ≈ 1
        @test @inferred(norm(q[i].(x),p)) ≈ 1
    end
    for i = 1:length(q), j = 1:length(q)
        @test abs(@inferred(dot(q[i]  ,q[j]  ,p)) - I[i,j]) <= sqrt(eps())
        @test abs(@inferred(dot(Q[:,i],q[j]  ,p)) - I[i,j]) <= sqrt(eps())
        @test abs(@inferred(dot(q[i]  ,Q[:,j],p)) - I[i,j]) <= sqrt(eps())
        @test abs(@inferred(dot(Q[:,i],Q[:,j],p)) - I[i,j]) <= sqrt(eps())
    end
    for i = 1:length(q)
        e = zeros(length(q)); e[i] = 1
        @test dot(q[i],q,p) ≈ e'
        @test dot(q[i],Q,p) ≈ e'
        @test dot(q[i],B,p) ≈ e'
        @test dot(Q[:,i],q,p) ≈ e'
        @test dot(Q[:,i],Q,p) ≈ e'
        @test dot(Q[:,i],B,p) ≈ e'
        @test dot(q,q[i],p) ≈ e
        @test dot(Q,q[i],p) ≈ e
        @test dot(B,q[i],p) ≈ e
        @test dot(q,Q[:,i],p) ≈ e
        @test dot(Q,Q[:,i],p) ≈ e
        @test dot(B,Q[:,i],p) ≈ e
    end
    @test dot(q,q,p) ≈ I
    @test dot(q,Q,p) ≈ I
    @test dot(q,B,p) ≈ I
    @test dot(Q,q,p) ≈ I
    @test dot(Q,Q,p) ≈ I
    @test dot(Q,B,p) ≈ I
    @test dot(B,q,p) ≈ I
    @test dot(B,Q,p) ≈ I
    @test dot(B,B,p) ≈ I
end
