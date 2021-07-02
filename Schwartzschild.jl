import Pkg; Pkg.activate(".")

using Geodesics
using DifferentialEquations
using Plots
using StaticArrays

## Choose graphics
#theme(:juno)        # you need to have PlotThemes installed
plotly()

## Choose metric
G(t,x,y,z) = Schwartzschild_metric(t,x,y,z;M=1);

## Options
# Maximum integration
Λ_max = 1000.0
# Maximum redshift
Z_max = 100.0;
# Initial position
ρs = 0.5;
x_u = [0., 0., 0., (2. + √3)*ρs];
# Initial direction
Ω = [1., 0., 0.];
Ω /= Ω'*Ω;

## Assemble initial conditions
# Create a null vector
e0_u, e1_u, e2_u, e3_u = make_tetrad(G, x_u...);
p_u = e0_u + Ω[1]*e1_u + Ω[2]*e2_u + Ω[3]*e3_u;
# Initial conditions
u0 = MVector{8}(vcat(x_u, p_u));

## Integrate equations
prob = ODEProblem(geodesic_equation(G), u0, (0, Λ_max));
integr = init(prob, Tsit5(), reltol=1e-8, abstol=1e-6)
while true
    step!(integr)
    if integr.t >= Λ_max
        break
    end
    if integr.u[5]/u0[5] > Z_max
        break
    end
end

## Plot the solution
plot(integr.sol, vars=(2,4))
