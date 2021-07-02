import Pkg; Pkg.activate(".")

using Geodesics
using DifferentialEquations
using Plots
using StaticArrays

# Choose metric
G(t,x,y,z) = Schwartzschild_metric(t,x,y,z;M=1);

# Initial position
ρs = 0.5;
x_u = [0., 0., 0., (2. + √3)*ρs];

# Direction
Ω = [1., 0., 0.];
Ω /= Ω'*Ω;

# Create a null vector
e0_u, e1_u, e2_u, e3_u = make_tetrad(G, x_u...);
p_u = e0_u + Ω[1]*e1_u + Ω[2]*e2_u + Ω[3]*e3_u;

# Solve differential equation
u0 = MVector{8}(vcat(x_u, p_u));
prob = ODEProblem(geodesic_equation(G), u0, (0., 100.));
@time prob = ODEProblem(geodesic_equation(G), u0, (0., 100.));
sol = solve(prob, Tsit5());

# Plot the solution
plot(sol, vars=(2,3,4))
