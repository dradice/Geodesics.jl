module Geodesics

using ForwardDiff
using Tullio
using LoopVectorization

export IT, IX, IY, IZ
export Schwartzschild_metric, KerrSchild_metric
export metric_deriv, coordinate_basis, make_tetrad, Christoffel_symbols
export geodesic_equation, geodesic_equation!

const IT = 1
const IX = 2
const IY = 3
const IZ = 4

"Schwartzschild metric in isotropic Cartesian coordinates"
function Schwartzschild_metric(t::Real, x::Real, y::Real, z::Real; M::Real = 1.0)
    ρs = M/2
    ρ = sqrt(x^2 + y^2 + z^2)
    g_dd = zeros(typeof(t), 4, 4)
    @inbounds g_dd[IT,IT] = -((1.0 - ρs/ρ)/(1.0 + ρs/ρ))^2
    for a = IX:IZ
        @inbounds g_dd[a,a] = (1.0 + ρs/ρ)^4
    end
    return g_dd
end

"Kerr metric in Cartesian Kerr-Schild coordinates"
function KerrSchild_metric(t::Real, x::Real, y::Real, z::Real; a::Real = 0.0, M::Real = 1.0)
    η_dd = zeros(typeof(t), 4, 4)
    @inbounds η_dd[IT, IT] = -1
    for a = IX:IZ
        @inbounds η_dd[a,a] = 1
    end
    # r is the implicit solution off
    # . (x^2 + y^2)/(r^2 + a^2) + z^2/r^2 = 1
    r = sqrt(-a^2/2 + x^2/2 + y^2/2 + z^2/2 + sqrt(a^4 - 2*a^2*x^2 - 2*a^2*y^2 + 2*a^2*z^2 + x^4 + 2*x^2*y^2 + 2*x^2*z^2 + y^4 + 2*y^2*z^2 + z^4)/2)
    f = 2*M*r^3/(r^4 + a^2*z^2)
    k_d = [1, (r*x + a*y)/(r^2 + a^2), (r*y - a*x)/(r^2 + a^2), z/r]
    @tullio g_dd[a,b] := η_dd[a,b] + f*k_d[a]*k_d[b]
    return g_dd
end

"""
Computes the derivative of the metric

. G must be a function returning a Matrix{Real,4} and taking t, x, y, z as input
"""
function metric_deriv(G, t::Real, x::Real, y::Real, z::Real)
    f(x_u) = vec(G(x_u...))
    Jac = ForwardDiff.jacobian(f, [t,x,y,z])
    dg_ddd = zeros(typeof(t), 4, 4, 4)
    for a = IT:IZ
        @inbounds dg_ddd[a,:,:] = reshape(Jac[:,a], 4, 4)
    end
    return dg_ddd
end

"Creates a coordinate basis (returned as tupled of 4 vectors)"
function coordinate_basis()
    ∂t_u = [1., 0., 0., 0.]
    ∂x_u = [0., 1., 0., 0.]
    ∂y_u = [0., 0., 1., 0.]
    ∂z_u = [0., 0., 0., 1.]
    return (∂t_u, ∂x_u, ∂y_u, ∂z_u)
end

function project_orthogonal(g_dd::Matrix, v::Vector, basis...)
    Pv = copy(v)
    for e in basis
        Pv -= (e'*g_dd*e)*(v'*g_dd*e)*e
    end
    return Pv
end

function normalize(g_dd::Matrix, v::Vector)
    v2 = v'*g_dd*v
    if v2 < 0
        return v/sqrt(-v2)
    elseif v2 > 0
        return v/sqrt(v2)
    end
    return v
end

"""
Create a tetrad using the Gram-Schmidt orthonormalization procedure

. g_dd must be a pseudo-Riemannian metric
. u, e1, e2, e3 must be linearly independent vectors
"""
function make_tetrad(g_dd::Matrix, u::Vector, e1::Vector, e2::Vector, e3::Vector)
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
Creates a tetrad using the coordinate basis as building block

. G must be a function returning a Matrix{Real,4} and taking t, x, y, z as input
"""
function make_tetrad(G, t::Real, x::Real, y::Real, z::Real)
    return make_tetrad(G(t, x, y, z), coordinate_basis()...)
end

"Computes the Christoffel symbols given the metric"
function Christoffel_symbols(g_uu::Matrix, dg_ddd::Array{T, 3} where T)
    @tullio Γ_udd[a,b,c] := 0.5*g_uu[a,d]*(dg_ddd[c,d,b] + dg_ddd[b,d,c] - dg_ddd[d,b,c])
    return Γ_udd
end

"""
Computes the Christoffel symbols given the metric function and the coordinates

. G must be a function returning a Matrix{Real,4} and taking t, x, y, z as input
"""
function Christoffel_symbols(G, t::Real, x::Real, y::Real, z::Real)
    dg_ddd = metric_deriv(G, t, x, y, z)
    g_uu = inv(G(t, x, y, z))
    return Christoffel_symbols(g_uu, dg_ddd)
end

"""
RHS for the geodesic equations (to be handed over to DifferentialEquations)
"""
function geodesic_equation(u::Vector, G, λ::Real)
    du = copy(u)
    geodesic_equation!(du, u, G, λ)
    return du
end

"""
RHS for the geodesic equations (to be handed over to DifferentialEquations)

This is the in place version
"""
function geodesic_equation!(du::Vector{T}, u::Vector{T}, G, λ::T) where T <: Real
    x_u = u[IT:IZ]
    p_u = u[IZ+1:end]
    Γ_udd = Christoffel_symbols(G, x_u...)
    dx_u = p_u
    @tullio dp_u[a] := - Γ_udd[a,b,c]*p_u[b]*p_u[c]
    du[IT:IZ] = dx_u[:]
    du[IZ+1:end] = dp_u[:]
end

end