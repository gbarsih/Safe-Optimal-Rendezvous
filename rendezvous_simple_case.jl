"""

 █████╗  ██████╗██████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██║
███████║██║     ██████╔╝██║
██╔══██║██║     ██╔══██╗██║
██║  ██║╚██████╗██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

File:       rendezvous_simple_case.jl
Author:     Gabriel Barsi Haberfeld, 2020. gbh2@illinois.edu
Function:   This program simulates all results in the paper "Geometry-Informed
            Minimum Risk Rendezvous Algorithm for Heterogeneous Agents in Urban
            Environments"

Instructions:   Run this file in juno with Julia 1.2.0 or later.
Requirements:   JuMP, Ipopt, Plots, LinearAlgebra, BenchmarkTools.

"""

using JuMP, Ipopt
using Plots, LinearAlgebra
default(dpi=600)
using BenchmarkTools
using Random
using RecipesBase
using SparseArrays
import Distributions: MvNormal
import Random.seed!
using SymPy
include("bayeslin.jl")

function solveRDV(x0,y0,Lx,Ly,Rx,Ry,vmin,vmax,tmax,rem_power,xp,yp)

    #Input variables:
    #   x0:         initial x
    #   xy:         initial y
    #   Lx:         Landing location x
    #   Ly:         Landing location y
    #   Rx:         Rendezvous location x
    #   Ry:         Rendezvous location y
    #   vmin:       minimum velocity
    #   vmax:       maximum velocity
    #   tmax:       max mission time
    #   rem_power:  battery remaining power
    #   xp:         previous x solution
    #   yp:         previous y solution

    RDV = Model(with_optimizer(Ipopt.Optimizer,max_iter=50,print_level=0))
    #RDV = Model(solver=CbcSolver(PrimalTolerance=1e-10))
    @variable(RDV, x[i=1:5]) #states
    @variable(RDV, y[i=1:5]) #states

    #x[1] : initial condition
    #x[2] : Point of no Return
    #x[3] : Rendezvous
    #x[4] : Landing
    #x[5] : PNR -> Landing

    @variable(RDV, vmin <= vx[i=1:4] <= vmax) #controls
    @variable(RDV, vmin <= vy[i=1:4] <= vmax) #controls

    #v[1] : Velocity up to Point of no Return
    #v[2] : V up to Rendezvous
    #v[3] : V up to landing
    #v[4] : PNR -> Landing

    @variable(RDV, 1.0 <= t[i=1:4]) #controls

    #@variable(RDV, 0 <= βx) #slack variables
    #@variable(RDV, 0 <= βy)

    @NLobjective(RDV, Min,
                  1.0*sum(vx[i]^2*t[i] + vy[i]^2*t[i] for i=1:2) #delivery
                + 0.5*(vx[3]^2*t[3] + vy[3]^2*t[3]) #cost fcn after delivery
                + 1.0*sum(t[i] for i=2:4)   #cost fcn min time
                - 1.0*t[1]) #cost fcn max decision time
                #+ 0.0*(βx + βy)) #slack minimization

    @constraint(RDV, x[1] == x0) #initial conditions
    @constraint(RDV, y[1] == y0)
    for i = 2:4
        @constraint(RDV, x[i] == x[i-1] + vx[i-1]*t[i-1]) #x Dynamics
        @constraint(RDV, y[i] == y[i-1] + vy[i-1]*t[i-1]) #y Dynamics
    end


    @constraint(RDV, x[5] == x[2] + vx[4]*t[4]) #Abort Dynamics Constraints
    @constraint(RDV, y[5] == y[2] + vy[4]*t[4])
    @NLconstraint(RDV,  (vx[1]^2 + vx[4]^2)*t[4] +
                        (vy[1]^2 + vy[4]^2)*t[4] <= rem_power) #TODO fix
    @constraint(RDV, x[5] == Lx)
    @constraint(RDV, y[5] == Ly)


    @NLconstraint(RDV, sum((vx[i]^2 + vy[i]^2)*t[i] for i=1:3) <= rem_power)
    #TODO fix
    θ_R = @NLexpression(RDV, sum(t[i] for i=1:2)/tmax)
    #@NLconstraint(RDV, c1, abs(x[3] - ( 5 + 0.5 * sin(2*pi*θ_R ))) <= βx)
    #@NLconstraint(RDV, c2, abs(y[3] - ( 2 * θ_R - 1 )) <= βy)
    #TODO see JuMP documentation to check constraint satisfaction
    # Problem:  adding NL constraint to x AND y makes the problem very hard
    # Solution: constraints satisfied with slack variables
    #@constraint(RDV, x[3] == Rx) #fixed rdv location
    #@constraint(RDV, y[3] == Ry)
    @NLconstraint(RDV, c1, x[3] == ( 5 + 4.5 * sin(2*pi*θ_R )))
    @NLconstraint(RDV, c2, y[3] == ( 2 * θ_R - 1 ))
    @NLconstraint(RDV, c3, θ_R <= 0.25)
    @constraint(RDV, x[4] == Lx)
    @constraint(RDV, y[4] == Ly)

    @constraint(RDV, sum(t[i] for i=1:4) <= tmax)

    optimize!(RDV)

    #β = [value.(βx) value.(βy)]
    #@show β
    θ_R = sum(value.(t)[1:2])/tmax

    return value.(x), value.(y), value.(vx), value.(vy), value.(t), θ_R

    #returns:
    #   x:      x waypoints
    #   y:      y waypoints
    #   vx:     velocities in x
    #   vy:     velocities in y
    #   t:      times to spend at those velocities

end

function path(θ)
    x = 5 .+ 4.5 .* sin.(2*pi*θ)
    y = 2 .* θ .- 1
    return x, y
end

θ̇(θ) = 0.1

function sim_path(θ̇, N, θ₀, tf)
    dt = tf/N
    θ = zeros(N)
    θ[1] = θ₀
    for i=2:N
        θ[i] = θ[i-1] + θ̇(θ[i-1])*dt
    end

    if maximum(θ) > 1.0 || minimum(θ) < 0.0
        @show maximum(θ) minimum(θ)
        error("Θ outside unit interval!")
    end

    return θ
end



function gen_A(σ,V,deg)
    Φ = basis_func(V,deg)
    A = σ^(-2) .* Φ * Φ' + I
end

function basis_func(V,deg)
    Φ = broadcast(^,V,collect(0:deg)')'
end

function μ_w(σ,V,deg,Y)
    A = gen_A(σ,V,deg)
    μ = 1/σ^2*inv(A)*basis_func(V,deg)*Y
end

function plot_path(n, bg="white")
    θ = Array(0:1.0/n:1)
    x, y = path(θ)
    plot!(x,y,background_color=bg)
end

function fit_behavior(N, α=0.005, β=1/(0.3^2), r=0:5; seeded=false)
    #D(v,β=Inf) = 0.0 + 1.1*v + 0.1*sin(1*pi*v) + 1/β*randn()
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    #deviation function, from θ̇.
    if seeded
        seed!(1729)
    end
    Xo = 2 ./3 * rand(N) .+ 1/3 #random samples
    Yo = D.(Xo, β) #observed deviation
    Xt = collect(-0.0:0.005:1.2)
    Yt = D.(Xt) #actual deviation
    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
    @show μ
    regress(Xo, Yo, Xt, Yt, polynomial, α, β, r)
end

function fit_weights(N, α=0.005, β=1/(0.3^2), r=0:5)
    D(v,β=Inf) = 5.0 + v + 1*sin(1.1*pi*v) + 1/β*randn()
    Xo = (2 .*rand(N) .- 1) .+ 40 #some velocity variation around 40MPH
    Yo = D.(Xo, β) #observed deviation
    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
end

function dynamics(x, y, vx, vy, tmax, dt, rem_power)
    x           = x + vx*dt
    y           = y + vy*dt
    tmax        = tmax - dt
    rem_power   = rem_power - vx^2*dt + vy^2*dt #TODO check if this is correct
    return x, y, tmax, rem_power
end

function run_fit()
    N = 50
    model = fit_behavior(N, 0.005, 1/(0.1^2), 0:3)
    plot(model, xlabel="Historic Speed", ylabel="Driver's Speed",background_color="black")
end

x0          = 0.0
y0          = 0.0
Lx          = 0.0
Ly          = 0.0
Rx          = 1.0
Ry          = 1.0
vmin        = -10.0
vmax        = 10.0
tmax        = 10.0
dt          = 0.1
rem_power   = 50.0

clearconsole()

#@btime solveRDV($x0,$y0,$Lx,$Ly,$Rx,$Ry,$vmin,$vmax,$tmax,$rem_power,$x0,$y0)
#@show bench

x, y, vx, vy, t, θ_R = @time solveRDV(x0,y0,Lx,Ly,Rx,Ry,vmin,vmax,tmax,rem_power,x0,y0)

@show θ_R

Σs = sum(sqrt.(x.^2+y.^2))
Σt = sum(t)
Δt = Σt - tmax
θ_R = sum(t[1:2])/tmax
p = path(θ_R)
Δx = abs(x[3] - p[1])
Δy = abs(y[3] - p[2])

println("Displaying resulting trajectory:")
@show x y vx vy t
println("Cheking Rendezvous condition")
@show Δx Δy
println("Checking constraints:")
@show Σs Σt Δt


plot(x[1:4],y[1:4],background_color="black")
scatter!(x[1:4],y[1:4],background_color="black")
plot_path(100,"black")
scatter!(p,legend=false,background_color="black")


run_fit()


"""
σ = 0.01
N = 1000
deg = 2
x = collect(1:N)
V = 1.0 .+ sin.(x/100.0)
Y = σ.^2 .*randn(N) .+ 0.5 + 2 .*cos.(x/100.0)

μ = μ_w(σ,V,deg,Y)
@show μ
A = gen_A(σ,V,deg)

T = zeros(N)

for i = 1:N
    temp = 0
    for j = 1:(deg+1)
        temp = temp + V[i]^(j-1)*μ[j]
    end
    T[i] = temp
end


plot(x,Y)
plot!(x,T)
#plot!(x,V)
"""
