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
using Printf
using LaTeXStrings
using Measures
import Distributions: MvNormal
import Random.seed!
include("bayeslin.jl")
default(dpi=100)

function solveRDV(x0,y0,t0,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,Σ,θ,N,xp,yp,tt=ones(4),vp=0.0)

    RDV = Model(with_optimizer(Ipopt.Optimizer,print_level=0,max_iter=500))
    @variable(RDV, x[i=1:5]) #states
    @variable(RDV, y[i=1:5]) #states

    @variable(RDV, -vmax <= vx[i=1:4] <= vmax) #controls
    @variable(RDV, -vmax <= vy[i=1:4] <= vmax) #controls
    @variable(RDV, 0.1 <= t[i=1:4]) #controls


    T_R = @expression(RDV, sum(t[i] for i=1:2))
    r = length(μ)
    t1 = t0
    t2 = @expression(RDV, t[2])
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

    Σv = @NLexpression(RDV, P1*(P1*Σ[1,1] + P2*Σ[2,1] + P3*Σ[3,1]) +
                            P2*(P1*Σ[1,2] + P2*Σ[2,2] + P3*Σ[3,2]) +
                            P3*(P1*Σ[1,3] + P2*Σ[2,3] + P3*Σ[3,3]))

    @NLobjective(RDV, Min,
                  1.0*sum(vx[i]^2*t[i] + vy[i]^2*t[i] for i=[1 2 4]) #delivery
                + 0.3*(vx[3]^2*t[3] + vy[3]^2*t[3]) #cost fcn after delivery
                + 0.0*sum(t[i] for i=2:4)   #cost fcn min time
                + 1.0*(N>=0)*Σv # activate risk min when samples are enough
                - 1.0*(N<=100)*t[1] #cost fcn max decision time
                - 0.0*t[1])
    @NLconstraint(RDV, abs(t[1] - tt[1]) <= 0.1)
    if xp != 0.0 || yp != 0.0
        @NLconstraint(RDV, abs(x[2] - xp) <= 0.01)
        @NLconstraint(RDV, abs(y[2] - yp) <= 0.01)
    end
    #@constraint(RDV, 2.0 .<= tt[2:end])
    #@NLconstraint(RDV, abs(vp[1] - vx[1]) <= 1.3)
    #@NLconstraint(RDV, abs(vp[2] - vy[1]) <= 1.3)
    @constraint(RDV, x[1] == x0) #initial conditions
    @constraint(RDV, y[1] == y0)
    for i = 2:4
        @constraint(RDV, x[i] == x[i-1] + vx[i-1]*t[i-1]) #x Dynamics
        @constraint(RDV, y[i] == y[i-1] + vy[i-1]*t[i-1]) #y Dynamics
    end

    @constraint(RDV, x[5] == x[2] + vx[4]*t[4]) #Abort Dynamics Constraints
    @constraint(RDV, y[5] == y[2] + vy[4]*t[4])
    @constraint(RDV, x[5] == Ax)
    @constraint(RDV, y[5] == Ay)


    @NLconstraint(RDV, sum((vx[i]^2 + vy[i]^2)*t[i] for i=1:3) <= rem_power)
    @NLconstraint(RDV, sum((vx[i]^2 + vy[i]^2)*t[i] for i=[1 4]) <= rem_power)


    if N >= 50 #activate risk bounds when enough samples
        @NLconstraint(RDV, Σv <= 2.5)
    else
        @NLconstraint(RDV, Σv <= 5.0)
    end
    @NLconstraint(RDV, θ_R <= 0.8)
    @NLconstraint(RDV, 0.2 + θ <= θ_R)
    @constraint(RDV, x[4] == Lx)
    @constraint(RDV, y[4] == Ly)
    #@NLconstraint(RDV, 0.01 <= vx[1]^2 + vy[1]^2)
    #@constraint(RDV, t[1] <= 10.0)
    @constraint(RDV, sum(t[i] for i=1:3) <= tmax)
    @constraint(RDV, sum(t[i] for i=[1 4]) <= tmax)
    @NLconstraint(RDV, x[3] == 5 - 4.5 * sin(2*pi*θ_R))
    @NLconstraint(RDV, y[3] == ( 10 * θ_R - 5 ))
    optimize!(RDV)

    t2 = sum(value.(t)[i] for i=1:2)
    t1 = t0

    θ_R = θ + μ[1]*(t2 - t1) +
        μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
        μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
        sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
        sin((4*t2*pi)/5)/16000)/pi)
    #@show θ_R

    return value.(x), value.(y), value.(vx), value.(vy), value.(t), θ_R
end

function path(θ)
    x = 5 .- 4.5 .* sin.(2*pi*θ)
    y = 10 .* θ .- 5
    return x, y
end

function fit_behavior(N, α=0.005, β=1/(0.3^2), r=0:2; seeded=false)
    #D(v,β=Inf) = 0.0 + 1.1*v + 0.1*sin(1*pi*v) + 1/β*randn()
    #deviation function, from θ̇.
    if seeded
        seed!(1729)
    end
    Xo = 0 .+ rand(N)./1.0 #random samples
    Yo = D.(Xo, β) #observed deviation
    Xt = collect(-0.0:0.005:1.2)
    Yt = D.(Xt) #actual deviation
    regress(Xo, Yo, Xt, Yt, polynomial, α, β, r)
end

function fit_weights(N, α=0.005, β=1/(0.3^2), r=0:2)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
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
    plot(model, xlabel="Historic Speed", ylabel="Driver's Speed",background_color="white",xlims=(0,1),ylims=(0,1.5))
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

function plot_sol(N=100,bg="black",t0=0.0,θ0=0.0,x0=0.0,y0=0.0)

    μ, Σ = fit_weights(N)
    θ̇(t) = (0.1 .+ 0.1.*cos.(4*pi.*t/10))

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
    tbound = 50.0
    tmax = 50.0
    tmax = find_t_end(μf,tmax,tbound,θ0,t0)
    #@show tmax
    x, y, vx, vy, t, θ_R = solveRDV(x0,y0,t0,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,Σ,θ0,N,x0,y0)
    T_R = sum(t[1:2])
    risk = Σv(t0,T_R)
    #@show θ_R T_R risk
    Σe = vx'.^2*t + vy'.^2*t
    Σs = sum(sqrt.(x.^2+y.^2))
    Σt = sum(t)
    Δt = Σt - tmax
    p = path(θ_R)
    Δx = abs(x[3] - p[1])
    Δy = abs(y[3] - p[2])
    @show x y vx vy t
    rR = Σv(t0,sum(t[1:2]))*(abs.(vx[1:2])'*t[1:2] + abs.(vy[1:2])'*t[1:2])/rem_power
    rA = Σv(t0,sum(t[1:2]))
    t1 = Float64(t[1])
    s = @sprintf("rho_R = %-15.4f\n rho_A = %-15.4f\n t1 = %-15.4f",rR,rA,t1)
    plot()
    plot!(path(collect(0:0.01:1)),color="gray",width=2.0)
    plot_path(1000,Σ,t0,θ0,tmax,bg,true)
    plot!(x[1:3],y[1:3],background_color=bg,width=3.0,color=:steelblue)
    plot!([x[2] ;x[5]],[y[2] ;y[5]],background_color=bg,width=1.0,color="gray",linestyle=:dash)
    plot!([x[3] ;x[4]],[y[3] ;y[4]],background_color=bg,width=1.0,color="gray",linestyle=:dash)
    scatter!([x[2]],[y[2]],background_color=bg,markersize=3.0,color=:grey)
    scatter!([x[1]],[y[1]],background_color=bg,markersize=7.0,markershape=:pentagon,color=:blue)
    scatter!([x[4]],[y[4]],background_color=bg,markersize=7.0,markershape=:utriangle,color=:orange)
    scatter!([x[5]],[y[5]],background_color=bg,markersize=7.0,markershape=:utriangle,color=:red)
    scatter!(path(θ0),background_color=bg,markersize=10.0,markershape=:square,color=:green,annotation=(2.5,3.0,s))
    p = scatter!(path(θ_R),legend=false,background_color=bg,markersize=7.0,xlims = (-1,11),ylims = (-6,6),color=:cyan)

    display(p)
    return vx[1], vy[1]
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

function plot_path(n,Σ,t0,θ0,tmax,bg="black",cgr=true)
    θ = Array(θ0:1.0/n:1)
    N = length(θ)
    x, y = path(θ)
    #z = θ̇.(collect(range(0,length=N,step=dt)))
    #z = z + D.(z)
    t0 = 0.0.*ones(N)
    t = collect(range(0,length=N,stop=tmax))
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
    P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
    sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
    sin((4*t2*pi)/5)/16000)/pi)
    Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
    z = Σv.(t0,t)
    #cgrad = cgrad([:red, :yellow, :blue])

    if cgr
        mcgrad = cgrad([:blue, :yellow, :red])
        plot!(x,y,background_color=bg,lc=mcgrad,line_z=z,width=3.0)
    else
        plot!(x,y,background_color=bg,width=3.0)
    end
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

function MPCfy(x0,y0,θ0,Lx,Ly,Ax,Ay,vmax,tmax,dt,Ni,H,rem_power,ρ=0.2,β = 1/(0.10^2))
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    θ̇(t) = (0.01 + 0.01*cos(4*pi*t/10))
    α = 0.001
    β = 1/(0.10^2)
    r = 0:2
    @assert 0<=ρ<=1 "FIR param error"
    tvec = zeros(H)
    μv = zeros(H)
    ρv = zeros(H)
    ρRv = zeros(H)
    x = zeros(H)
    y = zeros(H)
    td = zeros(H)
    x[1] = x0
    y[1] = y0
    xp = x0
    yp = y0
    t = 0
    Xo = rand(Ni) #random samples
    Yo = D.(Xo, β) #observed deviation
    μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
    μv[1] = μ[1]
    tt = 1*ones(4)
    vp = zeros(2)
    P1(t1,t2) = (t2 - t1)
    P2(t1,t2) = (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi))
    P3(t1,t2) = ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
    sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
    sin((4*t2*pi)/5)/16000)/pi)
    epsk = 0
    lx = 0.0
    ly = 4.0
    lvec = collect(range(ly, length=4, step=-0.8))
    k = 1
        anim = @animate for i = 2:H
            Xo = [Xo ; θ̇(θ0)]
            Yo = [Yo ; D(Xo[end], β)]
            μ0 = μ
            μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
            μ =  (1-ρ).*μ0 + ρ*μ
            μv[i] = μ[1]
            # ϕf(t1,t2) = [t2 - t1,
            #     (t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)),
            #     ((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 + sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 + sin((4*t2*pi)/5)/16000)/pi)]
            # Σf(t1,t2)=ϕf(t1,t2)'*Σ*ϕf(t1,t2)
            μf(t1,t2) = μ[1]*(t2 - t1) +
                μ[2]*(t2/100 - t1/100 - (sin((2*t1*pi)/5) - sin((2*t2*pi)/5))/(40*pi)) +
                μ[3]*((3*t2)/20000 - (3*t1)/20000 - (sin((2*t1*pi)/5)/2000 +
                sin((4*t1*pi)/5)/16000)/pi + (sin((2*t2*pi)/5)/2000 +
                sin((4*t2*pi)/5)/16000)/pi)

            Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                        P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                        P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
            tbound = 100.0
            tmax = 50.0
            tmax = find_t_end(μf,tmax,tbound,θ0,t)
            tvec[i-1] = tmax
            xt, yt, vx, vy, tt, θ_R = solveRDV(x[i-1],y[i-1],t,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,Σ,θ0,length(Xo),xp,yp,tt,vp)
            vp = [vx[1] vy[1]]
            xp = xt[2]
            yp = yt[2]
            Σe = vx'.^2*t + vy'.^2*t
            bg = "white"
            rA = Σv(t,sum(tt[1:2]))*(abs.(vx[1:2])'*tt[1:2] + abs.(vy[1:2])'*tt[1:2])/rem_power
            rR = Σv(t,sum(tt[1:2]))
            t1 = Float64(tt[1])
            if rR <= 0.01
                s1 = @sprintf("\$\\rho_R = %-15.4f\$",rR)
                s2 = @sprintf("\$\\rho_A = %-15.4f\$",rA)
                s3 = @sprintf("\$t_1= %-15.4f\$",t1)
                s4 = "Rendezvous is safe"
            else
                s1 = @sprintf("\$\\rho_R = %-15.4f\$",rR)
                s2 = @sprintf("\$\\rho_A = %-15.4f\$",rA)
                s3 = @sprintf("\$t_1= %-15.4f\$",t1)
                s4 = "Rendezvous is risky"
            end
            plot(path(collect(0:0.01:1)),color="gray",width=2.0)
            annotate!(lx,lvec[1], text(string(s1), 20, :left))
            annotate!(lx,lvec[2], text(string(s2), 20, :left))
            annotate!(lx,lvec[3], text(string(s3), 20, :left))
            annotate!(lx,lvec[4], text(string(s4), 20, (rR <= 0.01) ? :green : :red, :left))
            plot_path(1000,Σ,t0,θ0,tmax,bg,true)
            plot!(xt[1:3],yt[1:3],background_color=bg,width=3.0,color=:steelblue)
            plot!([xt[2] ;xt[5]],[yt[2] ;yt[5]],background_color=bg,width=1.0,color="gray",linestyle=:dash)
            plot!([xt[3] ;xt[4]],[yt[3] ;yt[4]],background_color=bg,width=1.0,color="gray",linestyle=:dash)
            scatter!([xt[2]],[yt[2]],background_color=bg,markersize=3.0,color=:grey)
            scatter!([xt[1]],[yt[1]],background_color=bg,markersize=7.0,markershape=:pentagon,color=:blue)
            scatter!([xt[4]],[yt[4]],background_color=bg,markersize=7.0,markershape=:utriangle,color=:orange)
            scatter!([xt[5]],[yt[5]],background_color=bg,markersize=7.0,markershape=:utriangle,color=:red)
            scatter!(path(θ0),background_color=bg,markersize=10.0,markershape=:square,color=:green)
            p = scatter!(path(θ_R),legend=false,background_color=bg,markersize=7.0,xlims = (-1,11),ylims = (-6,6),color=:cyan)
            p = plot!(tickfont = Plots.font("serif", pointsize=round(22.0)))
            if i in [3 127 233 310]
                println("PRINTING IMAGE")
                display(p)
                s = @sprintf("/Users/gabrielbarsi/Documents/GitHub/Safe-Optimal-Rendezvous/Safe-Optimal-Rendezvous/plot_%d.png",k)
                savefig(s)
                k = k + 1
            end
            xt, yt, rem_power, t = dynamics(xt[1], yt[1], vx[1], vy[1], t, dt, rem_power)
            x[i] = xt[1]
            y[i] = yt[1]
            ρ = Σv(t,sum(tt[1:2]))
            ρv[i-1] = ρ
            td[i-1] = tt[1]
            θ0 = θ0 + θ̇(θ0)*dt
            t_power = (abs.(vx[1:3])'*tt[1:3] + abs.(vy[1:3])'*tt[1:3])
            @show i
            # @show tt[1]
            # @show rem_power t_power tmax
            # @show θ0 θ_R vx vy ρ
            ρ_R = Σv(t,sum(tt[1:2]))*(abs.(vx[1:2])'*tt[1:2] + abs.(vy[1:2])'*tt[1:2])/rem_power
            # @show ρ_R
            ρRv[i-1] = ρ_R
            if ((tt[1] <= 1.0+1e-3) && (i>=10)) || (ρ_R >= 0.4)
                println("End Condition Met")
                epsk = i
                break
            end
        end#end for
        μ, Σ = posterior(Yo, polynomial(Xo, r), α, β)
        Σv(t1,t2) = P1(t1,t2)*(P1(t1,t2)*Σ[1,1] + P2(t1,t2)*Σ[2,1] + P3(t1,t2)*Σ[3,1]) +
                    P2(t1,t2)*(P1(t1,t2)*Σ[1,2] + P2(t1,t2)*Σ[2,2] + P3(t1,t2)*Σ[3,2]) +
                    P3(t1,t2)*(P1(t1,t2)*Σ[1,3] + P2(t1,t2)*Σ[2,3] + P3(t1,t2)*Σ[3,3])
        xt, yt, vx, vy, tt, θ_R = solveRDV(x[end],y[end],t,Lx,Ly,Ax,Ay,vmax,tmax,rem_power,μ,Σ,θ0,length(Xo),xp,yp)

        println("Assessing Risk:")
        ρ = Σv(t,sum(tt[1:2]))
        @show ρ
        if ρ<=0.01
            println("Mission is a go")
        else
            println("Start abort route")
        end
        tvec[end] = tvec[end-1]
        ρv[end] = ρv[end-1]
        ρRv[end] = ρRv[end-1]
        td[end] = td[end-1]
        gif(anim, "/Users/gabrielbarsi/Documents/GitHub/Safe-Optimal-Rendezvous/Safe-Optimal-Rendezvous/anim_fps30.gif", fps = 30)
        mp4(anim, "/Users/gabrielbarsi/Documents/GitHub/Safe-Optimal-Rendezvous/Safe-Optimal-Rendezvous/anim_fps30.mp4", fps = 30)
        return μv, tvec, ρv, td, ρRv, epsk
end

function genfigs(N,x0,y0,t0,θ0)
    D(v,β=Inf) = 2/(1+exp(-5*v)) - 1 + 1/β*randn()
    bg = "white"
    seed!(1729)
    yh = y0
    xh = x0
    vxh = 0.0
    vyh = 0.0
    dt = 0.1
    θ̇(t) = (0.1 .+ 0.1.*cos.(4*pi.*t/10))
    vx, vy = plot_sol(N,bg,t0,θ0,x0,y0)
    s = @sprintf("plot_%d.pdf",0)
    savefig(s)
    for i=1:20
        seed!(1729)
        s = @sprintf("plot_%d.pdf",i)
        t0+dt*i
        θ0 = θ0 + θ̇(t0)/1*dt
        x0 = x0+vx*dt*5
        y0 = y0+vy*dt*5
        vx, vy = plot_sol(N+10*i,bg,t0,θ0,x0,y0)
        xh = [xh x0]
        yh = [yh y0]
        vxh = [vxh vx]
        vyh = [vyh vy]
        savefig(s)
    end
    #plot(vyh')
    #p = plot!(yh')
    #display(p)
end

x0          = 10.0
y0          = -3.0
t0          = 0.0
θ0          = 0.0
# Lx          = 10.0
# Ly          = -y0 + 2
# Ax          = 7.5
# Ay          = -y0 + 2
Lx          = 5.0
Ly          = y0
Ax          = 7.5
Ay          = y0
vmax        = 2.5
tmax        = 10.0
dt          = 0.05
rem_power   = 20.0
N           = 1000
Ni          = 5
H           = Int(ceil(16/dt))

clearconsole()
#@benchmark solveRDV($x0,$y0,$t0,$Lx,$Ly,$Ax,$Ay,$vmax,$tmax,$rem_power,$μ,$Σ,$θ0,$N)

upscale = 1.0 #8x upscaling in resolution
fntsm = Plots.font("serif", pointsize=round(22.0*upscale))
fntlg = Plots.font("serif", pointsize=round(22.0*upscale))
default(bottom_margin=1mm)
default(top_margin=1mm)
default(right_margin=15mm)
default(left_margin=1mm)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)
default(size=(800*(upscale),600*(upscale))) #Plot canvas size
default(width = 3.0*upscale)

sn = 2
seed!(sn)
μfilt, tf, ρvf, tdf, ρRvf, epsk = MPCfy(x0,y0,θ0,Lx,Ly,Ax,Ay,vmax,tmax,dt,Ni,H,rem_power,0.2,1/(0.10^2))

upscale = 2.0 #8x upscaling in resolution
fntsm = Plots.font("serif", pointsize=round(14.0*upscale))
fntlg = Plots.font("serif", pointsize=round(14.0*upscale))
default(bottom_margin=15mm)
default(top_margin=5mm)
default(right_margin=5mm)
default(left_margin=20mm)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)
default(size=(800*(upscale),600*(upscale))) #Plot canvas size
default(width = 3.0*upscale)
plot()
plot!(ρvf,label=L"\rho_R")
plot!(ρRvf,label=L"\rho_A")
hline!([0.4],label=L"\gamma_A")
ylabel!("Risk")
p1 = hline!([0.01],label=L"\gamma_R")
p2 = plot(tdf,label=L"t_1")
xlabel!(L"k")
ylabel!(L"t_1[s]")
if epsk > 0
    p2 = vline!([epsk],label=L"t_1=\epsilon")
end
p = plot(p1,p2,layout=(2,1))
display(p)
savefig("risktimefail.pdf")




# t0 = t0+0.1
# θ0 = θ0+0.1
# plot_sol(N,"white",t0,θ0,x0,y0)
# genfigs(N,x0,y0,t0,θ0)

# seed!(sn)
# μnfilt, tu, ρvu, tdu, ρRvu = MPCfy(x0,y0,θ0,Lx,Ly,Ax,Ay,vmax,tmax,dt,Ni,H,rem_power,0.2,1/(0.30^2))
# p = plot(ρRvf,width=2.0)
# p = plot!(ρRvu,width=2.0)
# display(p)
# plot(ρvf,width=2.0)
# plot!(ρvu,width=2.0)
# p1 = hline!([0.01],width=2.0,ylims=(0.0,0.1))
# plot(tdf,width=2.0)
# plot!(tdu,width=2.0)
# p2 = hline!([0.1],width=2.0)
# plot(p1,p2,layout=(2,1))



# b = @benchmarkable solveRDV($x0,$y0,$t0,$Lx,$Ly,$Ax,$Ay,$vmax,$tmax,$rem_power,$μ,$Σ,$θ0,$N)
# tune!(b)
# run(b)
# plot(μfilt,label="filtered")
# plot!(μunfilt,label="raw")
#
# plot(tf,label="Filtered",width=2.0)
# plot!(tu,label="Raw",width=2.0)
# xlabel!("Control Iteration")
# ylabel!("Horizon Prediction")
# seed!(1729)
# plot_sol(N,"white",t0,θ0,x0,y0)

# seed!(1729)
# run_fit(50)
