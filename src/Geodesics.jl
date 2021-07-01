module Geodesics

using ForwardDiff
using StaticArrays, Tullio, LoopVectorization

export IT, IX, IY, IZ
export Schwartzschild_metric, KerrSchild_metric
export metric_deriv, coordinate_basis, make_tetrad, Christoffel_symbols
export geodesic_equation

const IT = 1
const IX = 2
const IY = 3
const IZ = 4

"""
    Schwartzschild_metric(t::Real, x::Real, y::Real, z::Real; M::Real = 1.0)

Computes the Schwartzschild spacetime metric in isotropic Cartesian coordinates [arXiv:0904.4184].
This function returns the metric as a matrix.
"""
function Schwartzschild_metric(t::Real, x::Real, y::Real, z::Real; M::Real = 1.0)
    ρs = M/2
    ρ = sqrt(x^2 + y^2 + z^2)
    g_dd = @MMatrix zeros(typeof(t), 4, 4)
    @inbounds g_dd[IT,IT] = -((1.0 - ρs/ρ)/(1.0 + ρs/ρ))^2
    for a = IX:IZ
        @inbounds g_dd[a,a] = (1.0 + ρs/ρ)^4
    end
    return SMatrix(g_dd)
end

"""
    KerrSchild_metric(t::Real, x::Real, y::Real, z::Real; a::Real = 0.0, M::Real = 1.0)

Computes the Kerr spacetime metric in Cartesian Kerr-Schild coordinates.
This function returns the metric as a matrix.
"""
function KerrSchild_metric(t::Real, x::Real, y::Real, z::Real; a::Real = 0.0, M::Real = 1.0)
    η_dd = @MMatrix zeros(typeof(t), 4, 4)
    @inbounds η_dd[IT, IT] = -1
    for a = IX:IZ
        @inbounds η_dd[a,a] = 1
    end
    # r is the implicit solution off
    # . (x^2 + y^2)/(r^2 + a^2) + z^2/r^2 = 1
    r = sqrt(-a^2/2 + x^2/2 + y^2/2 + z^2/2 + sqrt(a^4 - 2*a^2*x^2 - 2*a^2*y^2 + 2*a^2*z^2 + x^4 + 2*x^2*y^2 + 2*x^2*z^2 + y^4 + 2*y^2*z^2 + z^4)/2)
    f = 2*M*r^3/(r^4 + a^2*z^2)
    k_d = [1, (r*x + a*y)/(r^2 + a^2), (r*y - a*x)/(r^2 + a^2), z/r]
    g_dd = @MMatrix zeros(typeof(t), 4, 4)
    @tullio g_dd[a,b] = η_dd[a,b] + f*k_d[a]*k_d[b]
    return SMatrix(g_dd)
end

"""
    metric_deriv(G, t::Real, x::Real, y::Real, z::Real)

Computes the derivative of the metric

# Arguments
- `G` must be a function returning a `Matrix{Real,4}` and taking 4 `Real` as input
- `t`, `x`, `y`, and `z` are 4 `Real` specifing the spacetime point
"""
function metric_deriv(G, t::Real, x::Real, y::Real, z::Real)
    f(x_u) = vec(G(x_u...))
    Jac = ForwardDiff.jacobian(f, [t,x,y,z])
    dg_ddd = @MArray zeros(typeof(t), 4, 4, 4)
    for a = IT:IZ
        @inbounds dg_ddd[a,:,:] = reshape(Jac[:,a], 4, 4)
    end
    return SArray(dg_ddd)
end

"""
    coordinate_basis()

Creates a coordinate basis (returned as tupled of 4 vectors)
"""
function coordinate_basis()
    ∂t_u = @SVector [1., 0., 0., 0.]
    ∂x_u = @SVector [0., 1., 0., 0.]
    ∂y_u = @SVector [0., 0., 1., 0.]
    ∂z_u = @SVector [0., 0., 0., 1.]
    return (∂t_u, ∂x_u, ∂y_u, ∂z_u)
end

"Project a given vector ortogonally to a subspace spanned by a basis"
function project_orthogonal(g_dd::SMatrix, v::SVector, basis...)
    Pv = copy(v)
    for e in basis
        Pv -= (e'*g_dd*e)*(v'*g_dd*e)*e
    end
    return Pv
end

"Normalize a given vector"
function normalize(g_dd::SMatrix, v::SVector)
    v2 = v'*g_dd*v
    if v2 < 0
        return v/sqrt(-v2)
    elseif v2 > 0
        return v/sqrt(v2)
    end
    return v
end

"""
    make_tetrad(g_dd::Matrix, u::Vector, e1::Vector, e2::Vector, e3::Vector)

Create a tetrad using the Gram-Schmidt orthonormalization procedure

# Arguments
- `g_dd` must be a pseudo-Riemannian metric
- `u`, `e1`, `e2`, `e3` must be linearly independent vectors
"""
function make_tetrad(g_dd::SMatrix, u::SVector, e1::SVector, e2::SVector, e3::SVector)
    u = normalize(g_dd, u)

    e1 = project_orthogonal(g_dd, e1, u)
    e1 = normalize(g_dd, e1)

    e2 = project_orthogonal(g_dd, e2, u, e1)
    e2 = normalize(g_dd, e2)

    e3 = project_orthogonal(g_dd, e3, u, e1, e2)
    e3 = normalize(g_dd, e3)

    return (u, e1, e2, e3)
end

"""
    make_tetrad(G, t::Real, x::Real, y::Real, z::Real)

Creates a tetrad using the coordinate basis as building block

# Arguments
- `G` must be a function returning a `Matrix{Real,4}` and taking 4 `Real` as input
- `t`, `x`, `y`, and `z` are 4 `Real` specifing the spacetime point
"""
function make_tetrad(G, t::Real, x::Real, y::Real, z::Real)
    return make_tetrad(G(t, x, y, z), coordinate_basis()...)
end

"""
    Christoffel_symbols(g_uu::Matrix, dg_ddd::Array{T, 3} where T)

Computes the Christoffel symbols given the metric

# Arguments
- `g_uu` contravariant metric
- `dg_ddd` array of derivatives of the metric (as returned by `metric_deriv`)
"""
function Christoffel_symbols(g_uu::SMatrix, dg_ddd::SArray)
    Γ_udd = @MArray zeros(eltype(g_uu), 4, 4, 4)
    @tullio Γ_udd[a,b,c] = 0.5*g_uu[a,d]*(dg_ddd[c,d,b] + dg_ddd[b,d,c] - dg_ddd[d,b,c])
    return Γ_udd
end

"""
    Christoffel_symbols(G, t::Real, x::Real, y::Real, z::Real)

Computes the Christoffel symbols given the metric

# Arguments
- `G` must be a function returning a `Matrix{Real,4}` and taking 4 `Real` as input
- `t`, `x`, `y`, and `z` are 4 `Real` specifing the spacetime point
"""
function Christoffel_symbols(G, t::Real, x::Real, y::Real, z::Real)
    dg_ddd = metric_deriv(G, t, x, y, z)
    g_uu = inv(G(t, x, y, z))
    return Christoffel_symbols(g_uu, dg_ddd)
end

"""
    geodesic_equation(G)

Creates the RHS of the geodesic equation to be passed to `DifferentialEquations`

# Examples
```julia-repl
julia> metric(t, x, y, z) = Schwartzschild_metric(t, x, y, z, M=1.0);
julia> fun = geodesic_equation(metric);
julia> u0 = vcat(x0_u, p0_u);
julia> prob = ODEProblem(fun, u0, (0., 100.));
```
"""
function geodesic_equation(G)
    function fun!(du, u, p, λ)
        x_u = u[IT:IZ]
        p_u = u[IZ+1:end]
        Γ_udd = Christoffel_symbols(G, x_u...)
        du[IT:IZ] = p_u[:]
        dp = @MVector zeros(eltype(du), 4)
        @tullio dp[a] = - Γ_udd[a,b,c]*p_u[b]*p_u[c]
        du[IZ+1:end] = dp[:]
    end
    return fun!
end

end