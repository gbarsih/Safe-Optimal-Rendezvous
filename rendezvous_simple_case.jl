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
default(dpi=100)

function solveRDV(x0,y0,t0,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,θ,N)

    RDV = Model(with_optimizer(Ipopt.Optimizer,print_level=0,max_iter=1000))
    #RDV = Model(solver=CbcSolver(PrimalTolerance=1e-10))
    @variable(RDV, x[i=1:5]) #states
    @variable(RDV, y[i=1:5]) #states

    @variable(RDV, -vmax <= vx[i=1:4] <= vmax) #controls
    @variable(RDV, -vmax <= vy[i=1:4] <= vmax) #controls
    @variable(RDV, 0.1 <= t[i=1:4]) #controls


    T_R = @expression(RDV, sum(t[i] for i=1:2))
    r = length(μ)
    t1 = t0
    t2 = @expression(RDV, t[2])

    #[t2 - t1,
    #t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi),
    #(3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi]

    θ_R = @NLexpression(RDV, θ + μ[1]*((t[1] + t[2]) - t1) +
        μ[2]*((t[1] + t[2])/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*(t[1]
        + t[2])*pi)/5))/(40*pi)) +
        μ[3]*((3*(t[1] + t[2]))/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000
        + sin((4*t1*pi)/5)/16000)/pi + (sin((2*(t[1] + t[2])*pi)/5)/2000
        + sin((4*(t[1] + t[2])*pi)/5)/16000)/pi))
    P1 = @NLexpression(RDV, (t[1] + t[2]) - t1)
    P2 = @NLexpression(RDV, ((t[1] + t[2])/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*(t[1]
    + t[2])*pi)/5))/(40*pi)))
    P3 = @NLexpression(RDV, ((3*(t[1] + t[2]))/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000
    + sin((4*t1*pi)/5)/16000)/pi + (sin((2*(t[1] + t[2])*pi)/5)/2000
    + sin((4*(t[1] + t[2])*pi)/5)/16000)/pi))

    #=θ_R = @NLexpression(RDV, θ + μ[1]*((t[1] + t[2]) - t1) +
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
        t[2])*pi)/5)/640)/pi)=#

    Σv = @NLexpression(RDV, P1*(P1*Σ[1,1] + P2*Σ[2,1] + P3*Σ[3,1]) +
                            P2*(P1*Σ[1,2] + P2*Σ[2,2] + P3*Σ[3,2]) +
                            P3*(P1*Σ[1,3] + P2*Σ[2,3] + P3*Σ[3,3]))

    @NLobjective(RDV, Min,
                  1.0*sum(vx[i]^2*t[i] + vy[i]^2*t[i] for i=[1 2 4]) #delivery
                + 0.3*(vx[3]^2*t[3] + vy[3]^2*t[3]) #cost fcn after delivery
                + 0.0*sum(t[i] for i=2:4)   #cost fcn min time
                + 1.0*(N>=100)*Σv # activate risk min when samples are enough
                - 1.0*(Σv<=0.007)*t[1]) #cost fcn max decision time

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
    @constraint(RDV, x[5] == Ax)
    @constraint(RDV, y[5] == Ay)


    @NLconstraint(RDV, sum((vx[i]^2 + vy[i]^2)*t[i] for i=1:3) <= rem_power)
    @NLconstraint(RDV, sum((vx[i]^2 + vy[i]^2)*t[i] for i=[1 4]) <= rem_power)
    #TODO fix


    #@NLconstraint(RDV, x[3] ==
    #    5 + 4.5 * sin(2*pi*(θ + (sum(t[i] for i=1:2)
    #    - t0)*0.1 + (sum(t[i] for i=1:2)
    #    - t0)*sum(μ[i]*0.1^(i-1) for i=1:r))))
    #@NLconstraint(RDV, sum(t[i] for i=1:2)^2 * Σint <= 0.1)
    if N >= 50 #activate risk bounds when enough samples
        @NLconstraint(RDV, Σv <= 2.5)
    else
        @NLconstraint(RDV, Σv <= 5.0)
    end
    @NLconstraint(RDV, θ_R <= 0.8)
    @NLconstraint(RDV, 0.2 + θ <= θ_R)
    @constraint(RDV, x[4] == Lx)
    @constraint(RDV, y[4] == Ly)
    @constraint(RDV, sum(t[i] for i=1:3) <= tmax)
    @constraint(RDV, sum(t[i] for i=[1 4]) <= tmax)
    @NLconstraint(RDV, x[3] == 5 - 4.5 * sin(2*pi*θ_R))
    @NLconstraint(RDV, y[3] == ( 10 * θ_R - 5 ))
    optimize!(RDV)

    t2 = sum(value.(t)[i] for i=1:2)
    t1 = t0

    #[t2 - t1,
    #t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi),
    #(3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi]
    θ_R = θ + μ[1]*(t2 - t1) +
        μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
        μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
        sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
        sin((4*t2*pi)/5)/16000)/pi)
    @show θ_R
    #=θ_R = θ + μ[1]*(t2 - t1) +
        μ[2]*(t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi)) +
        μ[3]*((9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40
        + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40
        + sin((4*t2*pi)/5)/640)/pi)=#

    return value.(x), value.(y), value.(vx), value.(vy), value.(t), θ_R
end

function path(θ)
    x = 5 .- 4.5 .* sin.(2*pi*θ)
    y = 10 .* θ .- 5
    return x, y
end

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

function fit_behavior(N, α=0.005, β=1/(0.3^2), r=0:2; seeded=false)
    #D(v,β=Inf) = 0.0 + 1.1*v + 0.1*sin(1*pi*v) + 1/β*randn()
    #deviation function, from θ̇.
    if seeded
        seed!(1729)
    end
    Xo = 1/3 .+ rand(N)./2 #random samples
    Yo = D.(Xo, β) #observed deviation
    Xt = collect(-0.0:0.005:1.2)
    Yt = D.(Xt) #actual deviation
    regress(Xo, Yo, Xt, Yt, polynomial, α, β, r)
end

function fit_weights(N, α=0.005, β=1/(0.3^2), r=0:2)
    Xo = rand(N) #random samples
    Yo = D.(Xo, β) #observed deviation
    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
end

function fit_weights_filtered(N, μ0, α=0.005, β=1/(0.3^2), r=0:2)
    Xo = rand(N) #random samples
    Yo = D.(Xo, β) #observed deviation
    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
    μ = 0.2.*μ + 0.8.*μ0
    return μ, Σ
end

function dynamics(x, y, vx, vy, t, dt, rem_power)
    x           = x + vx*dt
    y           = y + vy*dt
    t           = t + dt
    rem_power   = rem_power - vx^2*dt - vy^2*dt #TODO check if this is correct
    return x, y, rem_power, t
end

function run_fit(N)
    model = fit_behavior(N, 0.005, 1/(0.3^2), 0:2)
    plot(model, xlabel="Historic Speed", ylabel="Driver's Speed",background_color="black")
end

function find_t_end(μf,tmax,tbound,θ0=0.0,t0=0.0)
    for c=t0:0.001:tbound
        if θ0 + μf(t0,c) >= 1.0
            return c - t0
        end#end if
    end#end for
    println("Couldn't find end of trajectory, refitting")
    return tmax
end#end fun

function plot_sol(N=100,bg="black")

    μ, Σ = fit_weights(N)
    θ̇(t) = (0.1 + 0.05*cos(4*pi*t/10))
    #=ϕf(t1,t2) = [t2 - t1,
        t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi),
        (9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40 + sin((4*t2*pi)/5)/640)/pi]
    Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
    μf(t1,t2) = μ[1]*(t2 - t1) +
        μ[2]*(t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi)) +
        μ[3]*((9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40 + sin((4*t2*pi)/5)/640)/pi)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2)/10 - t1/10 - (sin((2*t1*pi)/5) -
        sin((2*(t2)*pi)/5))/(8*pi)
    P3(t1,t2) = (9*(t2))/800 - (9*t1)/800 -
        (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi +
        (sin((2*(t2)*pi)/5)/40 + sin((4*(t2)*pi)/5)/640)/pi

    =#

    ϕf(t1,t2) = [t2 - t1,
        (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)),
        ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi)]
    Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
    μf(t1,t2) = μ[1]*(t2 - t1) +
        μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
        μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
        sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
        sin((4*t2*pi)/5)/16000)/pi)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
    P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
    sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
    sin((4*t2*pi)/5)/16000)/pi)

    Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
    tbound = 1000.0
    tmax = 1000.0
    tmax = find_t_end(μf,tmax,tbound)
    @show tmax
    x, y, vx, vy, t, θ_R = @time solveRDV(x0,y0,t0,Lx,Ly,vmax,tmax,rem_power,μ,θ0,N)
    T_R = sum(t[1:2])
    risk = Σv(t0,T_R)
    @show θ_R T_R risk
    Σe = vx'.^2*t + vy'.^2*t
    Σs = sum(sqrt.(x.^2+y.^2))
    Σt = sum(t)
    Δt = Σt - tmax
    p = path(θ_R)
    Δx = abs(x[3] - p[1])
    Δy = abs(y[3] - p[2])

    println("Displaying resulting trajectory:")
    @show x y vx vy t
    println("Checking Rendezvous condition")
    @show Δx Δy
    println("Checking constraints:")
    @show Σe Σs Σt Δt tmax

    plot(x[1:4],y[1:4],background_color=bg,width=3.0)
    plot!([x[2];x[5]],[y[2];y[5]],background_color=bg,width=3.0,color="red")
    plot_path(1000,bg)
    scatter!(x[1:4],y[1:4],background_color=bg,markersize=10.0)
    scatter!(p,legend=false,background_color=bg,markersize=5.0)
end

function plot_sol_filtered(μ0,N=100,bg="black")

    μ, Σ = fit_weights_filtered(N,μ0)
    θ̇(t) = (0.1 + 0.05*cos(4*pi*t/10))
    #=ϕf(t1,t2) = [t2 - t1,
        t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi),
        (9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40 + sin((4*t2*pi)/5)/640)/pi]
    Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
    μf(t1,t2) = μ[1]*(t2 - t1) +
        μ[2]*(t2/10 - t1/10 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(8*pi)) +
        μ[3]*((9*t2)/800 - (9*t1)/800 - (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi + (sin((2*t2*pi)/5)/40 + sin((4*t2*pi)/5)/640)/pi)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2)/10 - t1/10 - (sin((2*t1*pi)/5) -
        sin((2*(t2)*pi)/5))/(8*pi)
    P3(t1,t2) = (9*(t2))/800 - (9*t1)/800 -
        (sin((2*t1*pi)/5)/40 + sin((4*t1*pi)/5)/640)/pi +
        (sin((2*(t2)*pi)/5)/40 + sin((4*(t2)*pi)/5)/640)/pi

    =#

    ϕf(t1,t2) = [t2 - t1,
        (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)),
        ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi)]
    Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
    μf(t1,t2) = μ[1]*(t2 - t1) +
        μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
        μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
        sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
        sin((4*t2*pi)/5)/16000)/pi)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
    P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
    sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
    sin((4*t2*pi)/5)/16000)/pi)

    Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
    tbound = 1000.0
    tmax = 1000.0
    tmax = find_t_end(μf,tmax,tbound)
    @show tmax
    x, y, vx, vy, t, θ_R = @time solveRDV(x0,y0,t0,Lx,Ly,vmax,tmax,rem_power,μ,θ0,N)
    T_R = sum(t[1:2])
    risk = Σv(t0,T_R)
    @show θ_R T_R risk
    Σe = vx'.^2*t + vy'.^2*t
    Σs = sum(sqrt.(x.^2+y.^2))
    Σt = sum(t)
    Δt = Σt - tmax
    p = path(θ_R)
    Δx = abs(x[3] - p[1])
    Δy = abs(y[3] - p[2])

    println("Displaying resulting trajectory:")
    @show x y vx vy t
    println("Checking Rendezvous condition")
    @show Δx Δy
    println("Checking constraints:")
    @show Σe Σs Σt Δt tmax

    plot(x[1:4],y[1:4],background_color=bg,width=3.0)
    plot!([x[2];x[5]],[y[2];y[5]],background_color=bg,width=3.0,color="red")
    plot_path(1000,bg)
    scatter!(x[1:4],y[1:4],background_color=bg,markersize=10.0)
    p = scatter!(p,legend=false,background_color=bg,markersize=5.0)
    display(p)
    return μ
end

function plot_path(n,Σv,t0,θ0,tmax,bg="black")
    θ = Array(θ0:1.0/n:1)
    N = length(θ)
    x, y = path(θ)
    #z = θ̇.(collect(range(0,length=N,step=dt)))
    #z = z + D.(z)
    t0 = 0.0.*ones(N)
    t = collect(range(0,length=N,stop=tmax))
    z = Σv.(t0,t)
    #cgrad = cgrad([:red, :yellow, :blue])
    mcgrad = cgrad([:blue, :yellow, :red])
    plot!(x,y,background_color=bg,lc=mcgrad,line_z=z,width=3.0)
end

function plot_var(tf,N=100,bg="black")
    μ, Σ = fit_weights(N)
    ϕf(t1,t2) = [t2 - t1,
        (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)),
        ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi)]
    Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
    μf(t1,t2) = μ[1]*(t2 - t1) +
        μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
        μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
        sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
        sin((4*t2*pi)/5)/16000)/pi)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
    P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
    sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
    sin((4*t2*pi)/5)/16000)/pi)

    Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
    tbound = 3000.0
    tmax = tf
    tmax = find_t_end(μf,tmax,tbound)
    t = collect(range(0.0, length=N, stop=tmax))

    plot(t,Σv.(0.0,t),width=3.0,background_color=bg)
end

function MPCfy(x0,y0,θ0,Lx,Ly,vmax,tmax,dt,Ni,H,rem_power)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    θ̇(t) = (0.01 + 0.01*cos(4*pi*t/10))
    α = 0.005
    β = 1/(0.1^2)
    r = 0:2
    ρ = 0.2
    @assert 0<=ρ<=1 "FIR param error"

    x = zeros(H)
    y = zeros(H)
    x[1] = x0
    y[1] = y0
    t = 0
    Xo = rand(Ni) #random samples
    Yo = D.(Xo, β) #observed deviation

    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)

        anim = @animate for i = 2:H
            Xo = [Xo ; θ̇(θ0)]
            Yo = [Yo ; D(Xo[end], β)]
            μ0 = μ
            μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
            μ =  (1-ρ).*μ0 + ρ*μ

            ϕf(t1,t2) = [t2 - t1,
                (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)),
                ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi)]
            Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
            μf(t1,t2) = μ[1]*(t2 - t1) +
                μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
                μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
                sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
                sin((4*t2*pi)/5)/16000)/pi)
            P1(t1,t2) = (t2 - t1)
            P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
            P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
            sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
            sin((4*t2*pi)/5)/16000)/pi)

            Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                        P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                        P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
            tbound = 1000.0
            tmax = 50.0
            tmax = find_t_end(μf,tmax,tbound,θ0,t)
            xt, yt, vx, vy, tt, θ_R = solveRDV(x[i-1],y[i-1],t,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,θ0,length(Xo))
            Σe = vx'.^2*t + vy'.^2*t
            bg = "black"
            plot(xt[1:4],yt[1:4],background_color=bg,width=3.0)
            plot!([xt[2];xt[5]],[yt[2];yt[5]],background_color=bg,width=3.0,color="red")
            plot_path(1000,Σv,t0,θ0,tmax,bg)
            scatter!(xt[1:4],yt[1:4],background_color=bg,markersize=10.0)
            scatter!(path(θ0),background_color=bg,markersize=10.0,markershape=:star5)
            p = scatter!(path(θ_R),legend=false,background_color=bg,markersize=5.0,xlims = (0,10),ylims = (-5,5))
            #display(p)
            xt, yt, rem_power, t = dynamics(xt[1], yt[1], vx[1], vy[1], t, dt, rem_power)
            x[i] = xt[1]
            y[i] = yt[1]
            θ0 = θ0 + θ̇(θ0)*dt
            @show i
            @show tt[1]
            @show rem_power tmax
            @show θ0 vx vy Σv(t,sum(tt[1:2]))
            if (tt[1] <= 0.1+1e-3) || (θ_R <= θ0)
                println("End Condition Met")
                break
            end

        end#end for
        μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
        Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                    P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                    P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
        xt, yt, vx, vy, tt, θ_R = solveRDV(x[end],y[end],t,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,θ0,length(Xo))
        println("Assessing Risk:")
        ρ = Σv(t,sum(tt[1:2]))
        @show ρ
        if ρ<=0.004
            println("Mission is a go")
        else
            println("Start abort route")
        end
        gif(anim, "/Users/gabrielbarsi/Documents/GitHub/Safe-Optimal-Rendezvous/anim_fps30.gif", fps = 30)
end

x0          = 8.0
y0          = -3.0
t0          = 0.0
θ0          = 0.0
Lx          = 1.0
Ly          = -0.5
Ax          = 4.0
Ay          = -3.0
vmax        = 1.0
tmax        = 10.0
dt          = 0.1
rem_power   = 20.0
N           = 51
Ni          = 100
H           = Int(ceil(40/dt))

clearconsole()

MPCfy(x0,y0,θ0,Lx,Ly,vmax,tmax,dt,Ni,H,rem_power)
#=
#Σf(t) = (θf(t).^collect(1:length(μ))'*Σ*θf(t).^collect(1:length(μ)))[1]
μ, Σ = fit_weights(N)
ϕf(t1,t2) = [t2 - t1,
    (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)),
    ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi)]
Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
μf(t1,t2) = μ[1]*(t2 - t1) +
    μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
    μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
    sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
    sin((4*t2*pi)/5)/16000)/pi)
P1(t1,t2) = (t2 - t1)
P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
sin((4*t2*pi)/5)/16000)/pi)

Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
            P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
            P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
run_fit(10)
plot_sol(N)


#@btime solveRDV($x0,$y0,$t0,$Lx,$Ly,$Rx,$Ry,$vmax,$tmax,$rem_power,$μ,$Σint,$θ0,$N)
=#
