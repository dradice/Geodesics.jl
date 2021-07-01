using Geodesics
using Test

@testset "Geodesics.jl" begin
    metric(t,x,y,z) = Schwartzschild_metric(t, x, y, z, M=2.0)
    g_dd = metric(0., 0.5, 0.5, 0.5)
    tetrad = make_tetrad(g_dd, coordinate_basis()...)
    @test tetrad[1]'*g_dd*tetrad[1] ≈ -1.0
    for a = IX:IZ
        @test tetrad[a]'*g_dd*tetrad[a] ≈ 1.0
    end
    for a = IT:IZ
        for b = IT:IZ
            if b != a
                @test tetrad[a]'*g_dd*tetrad[b] ≈ 0.0
            end
        end
    end
end
