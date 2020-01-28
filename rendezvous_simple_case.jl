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
using BenchmarkTools
using Random
using RecipesBase
using SparseArrays
using Statistics
import Distributions: MvNormal
import Random.seed!
include("bayeslin.jl")
default(dpi=600)

function solveRDV(x0,y0,t0,Lx,Ly,Rx,Ry,vmax,tmax,rem_power,μ,Σint,θ)

    RDV = Model(with_optimizer(Ipopt.Optimizer,max_iter=1000))
    #RDV = Model(solver=CbcSolver(PrimalTolerance=1e-10))
    @variable(RDV, x[i=1:5]) #states
    @variable(RDV, y[i=1:5]) #states
    #x[1] : initial condition
    #x[2] : Point of no Return
    #x[3] : Rendezvous
    #x[4] : Landing
    #x[5] : PNR -> Landing

    @variable(RDV, -vmax <= vx[i=1:4] <= vmax) #controls
    @variable(RDV, -vmax <= vy[i=1:4] <= vmax) #controls

    #v[1] : Velocity up to Point of no Return
    #v[2] : V up to Rendezvous
    #v[3] : V up to landing
    #v[4] : PNR -> Landing

    @variable(RDV, 1.0 <= t[i=1:4]) #controls

    @NLobjective(RDV, Min,
                  1.0*sum(vx[i]^2*t[i] + vy[i]^2*t[i] for i=[1 2 4]) #delivery
                + 0.5*(vx[3]^2*t[3] + vy[3]^2*t[3]) #cost fcn after delivery
                + 1.0*sum(t[i] for i=2:4)   #cost fcn min time
                + 0.0*sum(t[i] for i=1:2)*Σint
                - 10.0*t[1]) #cost fcn max decision time

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
    T_R = @expression(RDV, sum(t[i] for i=1:2))
    #0.1 is θ̇(t)
    r = length(μ)
    #θ_R = @expression(RDV, θ + (T_R - t0)*0.1 + (T_R - t0)*sum(μ[i]*0.1^(i-1) for i=1:r))
    t1 = t0
    t2 = @expression(RDV, t[2])
    #=θ_R = @NLexpression(RDV, sin((9*sin(2*pi*θ
    - (μ[2]*sin((2*pi*t1)/5))/4
    + (μ[2]*sin((2*pi*t2)/5))/4
    - (μ[3]*sin((2*pi*t1)/5))/20
    + (μ[3]*sin((2*pi*t2)/5))/20
    - (μ[3]*sin((4*pi*t1)/5))/320
    + (μ[3]*sin((4*pi*t2)/5))/320
    - 2*pi*μ[1]*t1
    + 2*pi*μ[1]*t2
    - (pi*μ[2]*t1)/5
    + (pi*μ[2]*t2)/5
    - (9*pi*μ[3]*t1)/400
    + (9*pi*μ[3]*t2)/400))/2 + 5))=#
    #=θ_R = @NLexpression(RDV, θ - μ[1]*(t1 - t2)
    - μ[3]*((9*t1)/800 - (9*t2)/800 + (sin((2*pi*t1)/5)/40
    + sin((4*pi*t1)/5)/640)/pi - (sin((2*pi*t2)/5)/40
    + sin((4*pi*t2)/5)/640)/pi) - μ[2]*(t1/10 - t2/10
    + (sin((2*pi*t1)/5) - sin((2*pi*t2)/5))/(8*pi)))=#

    θ_R = @NLexpression(RDV, θ + μ[1]*((t[1] + t[2]) - t1) +
        μ[2]*((t[1] + t[2])/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*(t[1]
        + t[2])*pi)/5))/(8*pi)) +
        μ[3]*((9*(t[1] + t[2]))/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40
        + sin((4*t1*pi)/5)/640)/pi + (sin((2*(t[1] + t[2])*pi)/5)/40
        + sin((4*(t[1] + t[2])*pi)/5)/640)/pi))
    P1 = @NLexpression(RDV, (t[1] + t[2]) - t1)
    P2 = @NLexpression(RDV, (t[1] + t[2])/10 - t1/10 - (sin((2*t1*pi)/5) -
        sin((2*(t[1] + t[2])*pi)/5))/(8*pi))
    P3 = @NLexpression(RDV, (9*(t[1] + t[2]))/800 - (9*t1)/800 -
        (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi +
        (sin((2*(t[1] + t[2])*pi)/5)/40 + sin((4*(t[1] +
        t[2])*pi)/5)/640)/pi)

    Σv = @NLexpression(RDV, P1*(P1*Σ[1,1] +
                            P2*Σ[2,1] + P3*Σ[3,1]) +
                            P2*(P1*Σ[1,2] + P2*Σ[2,2] + P3*Σ[3,2]) +
                            P3*(P1*Σ[1,3] + P2*Σ[2,3] + P3*Σ[3,3]))

    #@NLconstraint(RDV, x[3] ==
    #    5 + 4.5 * sin(2*pi*(θ + (sum(t[i] for i=1:2)
    #    - t0)*0.1 + (sum(t[i] for i=1:2)
    #    - t0)*sum(μ[i]*0.1^(i-1) for i=1:r))))
    @NLconstraint(RDV, x[3] == 5 + 4.5 * sin(2*pi*θ_R))
    @NLconstraint(RDV, y[3] == ( 2 * θ_R - 1 ))
    #@NLconstraint(RDV, sum(t[i] for i=1:2)^2 * Σint <= 0.1)
    @NLconstraint(RDV, Σv <= 100000.15)
    @NLconstraint(RDV, θ_R <= 1.0)
    @NLconstraint(RDV, 0.0 <= θ_R)
    @constraint(RDV, x[4] == Lx)
    @constraint(RDV, y[4] == Ly)

    @constraint(RDV, sum(t[i] for i=1:4) <= tmax)

    optimize!(RDV)

    t2 = sum(value.(t)[i] for i=1:2)
    t1 = t0
    θ_R = θ + μ[1]*(t2 - t1) +
        μ[2]*(t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi)) +
        μ[3]*((9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40
        + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40
        + sin((4*t2*pi)/5)/640)/pi)

    return value.(x), value.(y), value.(vx), value.(vy), value.(t), θ_R

end

function path(θ)
    x = 5 .+ 4.5 .* sin.(2*pi*θ)
    y = 2 .* θ .- 1
    return x, y
end

#θ̇(t) = 0.1
θ̇(t) = (0.1 + 0.05*cos(4*pi*t/10))

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

function fit_behavior(N, α=0.005, β=1/(0.3^2), r=0:2; seeded=false)
    #D(v,β=Inf) = 0.0 + 1.1*v + 0.1*sin(1*pi*v) + 1/β*randn()
    D(v,β=Inf) = 2/(1+exp(-2*v)) - 1 + 1/β*randn()
    #deviation function, from θ̇.
    if seeded
        seed!(1729)
    end
    Xo = 2 ./3 * rand(N) .+ 0/3 #random samples
    Yo = D.(Xo, β) #observed deviation
    Xt = collect(-0.0:0.005:1.2)
    Yt = D.(Xt) #actual deviation
    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
    @show μ
    regress(Xo, Yo, Xt, Yt, polynomial, α, β, r)
end

function fit_weights(N, α=0.005, β=1/(0.3^2), r=0:2)
    D(v,β=Inf) = 2/(1+exp(-2*v)) - 1 + 1/β*randn()
    Xo = 2 ./3 * rand(N) .+ 0/3 #random samples
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
    N = 100
    model = fit_behavior(N, 0.005, 1/(0.3^2), 0:2)
    plot(model, xlabel="Historic Speed", ylabel="Driver's Speed",background_color="black")
end

function plot_sol(bg="black")
    plot(x[1:4],y[1:4],background_color=bg)
    scatter!(x[1:4],y[1:4],background_color=bg)
    plot_path(100,bg)
    scatter!(p,legend=false,background_color=bg)
end

x0          = 0.0
y0          = 0.0
t0          = 0.0
θ0          = 0.0
Lx          = 0.0
Ly          = 0.0
Rx          = 1.0
Ry          = 1.0
vmax        = 10.0
tmax        = 10.0
dt          = 0.1
rem_power   = 50000.0

clearconsole()

#@btime solveRDV($x0,$y0,$Lx,$Ly,$Rx,$Ry,$vmin,$vmax,$tmax,$rem_power,$x0,$y0)
#@show bench

μ, Σ = fit_weights(100)

θf(t) = 0.1 + sin(t)
Σint = (0.1.^collect(1:length(μ))'*Σ*0.1.^collect(1:length(μ)))[1]
#Σf(t) = (θf(t).^collect(1:length(μ))'*Σ*θf(t).^collect(1:length(μ)))[1]

"""
Functions for slow corner speeds.
"""
#ϕf(t1,t2) = [t2-t1
#    -0.1*t1-0.0397887*sin((2*π*t1)/5)+0.1*t2+0.0397887*sin((2*π*t2)/5)
#    -0.01125*t1-0.00795775*sin((2*π*t1)/5)-0.000497359*sin((4*π*t1)/5)+0.01125*t2+0.00795775*sin((2*π*t2)/5)+0.000497359*sin((4*π*t2)/5)]
#=ϕf(t1,t2) = [t2 - t1,
    t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi),
    (9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40
    + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40
    + sin((4*t2*pi)/5)/640)/pi]
    =#
ϕf(t1,t2) = [t2 - t1,
    t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi),
    (9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40 + sin((4*t2*pi)/5)/640)/pi]
Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)

μf(t1,t2) = μ[1]*(t2 - t1) +
    μ[2]*(t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi)) +
    μ[3]*((9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40 + sin((4*t2*pi)/5)/640)/pi)

x, y, vx, vy, t, θ_R = @time solveRDV(x0,y0,t0,Lx,Ly,Rx,Ry,vmax,tmax,rem_power,μ,Σint,θ0)

T_R = sum(t[1:2])
risk = Σf(t0,T_R)
@show θ_R T_R risk

Σs = sum(sqrt.(x.^2+y.^2))
Σt = sum(t)
Δt = Σt - tmax
p = path(θ_R)
Δx = abs(x[3] - p[1])
Δy = abs(y[3] - p[2])

println("Displaying resulting trajectory:")
@show x y vx vy t
println("Cheking Rendezvous condition")
@show Δx Δy
println("Checking constraints:")
@show Σs Σt Δt




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
