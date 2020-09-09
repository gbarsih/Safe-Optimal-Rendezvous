"""

 █████╗  ██████╗██████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██║
███████║██║     ██████╔╝██║
██╔══██║██║     ██╔══██╗██║
██║  ██║╚██████╗██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

File:       rendezvous_review.jl
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
default(dpi = 100)

function uav_dynamics(x, y, vx, vy, dt, rem_power = Inf, vmax = Inf, m = 1.0)
    vx > vmax ? vmax : vx
    vy > vmax ? vmax : vy
    vx < -vmax ? -vmax : vx
    vy < -vmax ? -vmax : vy
    x = x + vx * dt
    y = y + vy * dt
    rem_power = rem_power - vx^2 * m / 2 * dt - vy^2 * m / 2 * dt - 1*dt
    return x, y, rem_power
end

function driver_dynamics(θ, t, dt)
    return θ .+ speed_profile(t) .* dt
end

function path(θ)
    x = θ .* 1000
    y = θ .* 1000
    return x, y
end

function plot_path(θ0, n = 100, bg = "white", cgr = false)
    θ = Array(θ0:1.0/n:1)
    N = length(θ)
    x, y = path(θ)

    if cgr
        mcgrad = cgrad([:blue, :yellow, :red])
        plot!(x, y, background_color = bg, lc = mcgrad, line_z = z, width = 3.0)
    else
        plot!(x, y, background_color = bg, width = 3.0)
    end
end

function speed_profile(t)
    # This is the historical speed profile
    #1/100 .- 1/20000 .*t
    1/200
end

function position_profile(t)
    #-t^3 / 12000000 + t^2 / 40000 + t / 400 + 1 / 12
    #-(t.*(t .- 400))./40000
    t./200
end

function time_profile(θd)
    #200 - 20000*(1/10000 - θd/10000)^(1/2)
    200*θd
end

function animate_entities()
    n = 10
    t = Array(0:1/n:200)
    N = length(t)
    D = 1000 * sqrt(2)
    xdriver, ydriver = path(0)
    θ = 0
    x = 0
    y = 500
    p = 2000
    anim = @animate for i = 1:N
        plot(label = false)
        plot_path(0)
        scatter!(path(θ), markersize = 10.0, markershape = :square, color = :green)
        scatter!([x], [y], markersize = 10.0, markershape = :circle, color = :black)
        s1 = @sprintf("Driver's speed = %15.2f km/h", speed_profile(t[i]) * D * 3.6)
        annotate!(0, 1000, text(string(s1), 20, :left))
        θ = driver_dynamics(θ, t[i], 1 / n)
        θi = position_profile(t[i])
        x, y, p = uav_dynamics(x, y, 2.5, 0, 1 / n, p)
        @show θ θi x y p
    end every 50
    gif(anim, "driver_anim.gif", fps = 10)
end

function basicMPC(
    t0 = 0,
    x = [500, 0],
    L = [500, 0],
    dt = 1 / 10,
    T = 200,
    vmax = 10.0,
    p = 5000,
    random_profile = false,
)
    MPC = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter" => 50,
    ))
    @variable(MPC, PNR[i = 1:2])
    @variable(MPC, RDV[i = 1:2])
    @variable(MPC, -vmax <= vx[i = 1:4] <= vmax) #controls
    @variable(MPC, -vmax <= vy[i = 1:4] <= vmax) #controls
    @variable(MPC, 0.1 <= t[i = 1:4]) #controls
    t1 = t0
    t_R = @NLexpression(MPC, t[2] + t[1] + t0)
    θ_R = @NLexpression(MPC, -(t_R*(t_R - 400))/40000)
    @NLobjective(
        MPC,
        Min,
        1.0 * sum(vx[i]^2 * t[i] + vy[i]^2 * t[i] for i in [1 2 4]) +
        0.3 * (vx[3]^2 * t[3] + vy[3]^2 * t[3])
    )

    @constraint(MPC, PNR[1] == x[1] + vx[1] * t[1])
    @constraint(MPC, PNR[2] == x[2] + vy[1] * t[1])
    @constraint(MPC, RDV[1] == PNR[1] + vx[2] * t[2])
    @constraint(MPC, RDV[2] == PNR[2] + vy[2] * t[2])
    @constraint(MPC, L[1] == RDV[1] + vx[3] * t[3])
    @constraint(MPC, L[2] == RDV[2] + vy[3] * t[3])
    @constraint(MPC, L[1] == PNR[1] + vx[4] * t[4])
    @constraint(MPC, L[2] == PNR[2] + vy[4] * t[4])

    @NLconstraint(MPC, θ_R >= 0)
    @NLconstraint(MPC, θ_R <= 1)

    @NLconstraint(MPC, RDV[1] == θ_R * 1000)
    @NLconstraint(MPC, RDV[2] == θ_R * 1000)

    @constraint(MPC, t[1] + t[2] + t[3] <= T)
    @constraint(MPC, t[1] + t[2] + t[4] <= T)

    @NLconstraint(MPC, sum((vx[i]^2 + vy[i]^2) * t[i] for i = 1:3) <= p)
    @NLconstraint(MPC, (vx[1]^2 + vy[1]^2) * t[1] + (vx[1]^2 + vy[1]^2) * t[1] <= p)

    optimize!(MPC)

    return value.(RDV), value.(PNR), value.(t), value.(vx), value.(vy)
end

function deterministicMPC(
    θd,
    PNRp,
    vxp,
    vyp,
    θRp,
    t0 = 0.0,
    x = [500.0, 0.0],
    L = [500.0, 0.0],
    dt = 1.0,
    T = 200.0,
    vmax = Inf,
    P = 5000.0,
    random_profile = false,
)
    MPC = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter" => 500,
    ))
    @variable(MPC, PNR[i = 1:2])
    @variable(MPC, RDV[i = 1:2])
    @variable(MPC, -vmax <= vx[i = 1:4] <= vmax) #controls
    @variable(MPC, -vmax <= vy[i = 1:4] <= vmax) #controls
    @variable(MPC, 0.1 <= t[i = 1:4]) #controls
    t_i = time_profile(θd) #relative adjusted time
    t_R = @NLexpression(MPC, t[2] + t[1] + t0) #absolute rendezvous time
    t_a = @NLexpression(MPC, t_R - t0 + t_i) #adjusted rendezvous time
    #θ_R = @NLexpression(MPC, -(t_a*(t_a - 400))/40000) #rendezvous location
    θ_R = @NLexpression(MPC, t_a/200) #rendezvous location
    @NLobjective(
        MPC,
        Min,
        1.0 * sum(vx[i]^2 * t[i] + vy[i]^2 * t[i] for i in [1 2 4]) +
        1.0 * (vx[3]^2 * t[3] + vy[3]^2 * t[3])
    )

    r = 10
    sr = 0.3;
    tr = 0.3;

    @constraint(MPC, (PNRp[1] - PNR[1])^2 <= r^2)
    @constraint(MPC, (PNRp[2] - PNR[2])^2 <= r^2)
    for i=1:4
        @constraint(MPC, (vxp[i] - vx[i])^2 <= sr^2)
        @constraint(MPC, (vyp[i] - vy[i])^2 <= sr^2)
    end
    @NLconstraint(MPC, (θRp - θ_R)^2 <= tr^2)
    @constraint(MPC, PNR[1] == x[1] + vx[1] * t[1])
    @constraint(MPC, PNR[2] == x[2] + vy[1] * t[1])
    @constraint(MPC, RDV[1] == PNR[1] + vx[2] * t[2])
    @constraint(MPC, RDV[2] == PNR[2] + vy[2] * t[2])
    @constraint(MPC, L[1] == RDV[1] + vx[3] * t[3])
    @constraint(MPC, L[2] == RDV[2] + vy[3] * t[3])
    @constraint(MPC, L[1] == PNR[1] + vx[4]*t[4])
    @constraint(MPC, L[2] == PNR[2] + vy[4]*t[4])

    @NLconstraint(MPC, θ_R >= 0)
    @NLconstraint(MPC, θ_R <= 1)

    @NLconstraint(MPC, RDV[1] == θ_R * 1000)
    @NLconstraint(MPC, RDV[2] == θ_R * 1000)

    @constraint(MPC, t[1] + t[2] + t[3] <= T)
    @constraint(MPC, t[1] + t[2] + t[4] <= T)

    @NLconstraint(MPC, P >= sum((vx[i]^2 + vy[i]^2) * t[i] + 1*t[i] for i = 1:3))
    @NLconstraint(MPC, P >= sum((vx[i]^2 + vy[i]^2) * t[i] + 1*t[i] for i in [1 4]))

    optimize!(MPC)

    return value.(RDV), value.(PNR), value.(t), value.(vx), value.(vy)
end

function deterministicMission(
    x0 = [500.0, 0.0],
    L = [500.0, 0],
    P = 2700.0,
    T = 200.0,
    vmax = 10.0;
    dt = 1.0,
)
    tv = Array(0:dt:100)
    N = length(tv)
    distance = zeros(N)
    power = zeros(N)
    power_abort = zeros(N)
    power_return = zeros(N)
    power_rendezvous = zeros(N)
    θv = zeros(N)
    θd = 0.0
    θR = 0.5
    PNR = x0
    Vx = [0.0, 0.0, 0.0, 0.0]
    Vy = [0.0, 0.0, 0.0, 0.0]
    for i = 1:N
        RDV, PNR, t, Vx, Vy = deterministicMPC(θd, PNR, Vx, Vy, θR, tv[i], x0, L, dt, T, vmax, P)
        x = x0[1]
        y = x0[2]
        vx = Vx[1]
        vy = Vy[1]
        v = [vx vy]
        θR = position_profile(sum(t[1:2])+tv[i])
        x0[1], x0[2], P = uav_dynamics(x, y, vx, vy, dt, P)
        θd = position_profile(1.00*tv[i]) #driver following prototypical profile
        Pa = t[1]*(Vx[1]^2 + Vy[1]^2) + t[4]*(Vx[4]^2 + Vy[4]^2)
        Pr = t[3]*(Vx[3]^2 + Vy[3]^2)
        Pd = t[1]*(Vx[1]^2 + Vy[1]^2) + t[2]*(Vx[2]^2 + Vy[2]^2)
        distance[i] = norm(x0 .- path(θd))
        power[i] = P
        power_abort[i] = Pa
        power_return[i] = Pr
        power_rendezvous[i] = Pd
        θv[i] = θR
        cRDVx = Vx[1]*t[1] + Vx[2]*t[2] + x0[1]
        cRDVy = Vy[1]*t[1] + Vy[2]*t[2] + x0[2]
        cLx = Vx[1]*t[1] + Vx[2]*t[2] + Vx[3]*t[3] + x0[1]
        cLy = Vy[1]*t[1] + Vy[2]*t[2] + Vy[3]*t[3] + x0[2]
        @show tv[i] θd θR x0 RDV v P Pa Pr Pd cRDVx cRDVy cLx cLy
        if norm(x0 .- path(θd)) < 3.0
            println("Rendezvous Successful")
            break
        elseif (Pa>P+100 || Pr>P+100)
            println("Abort Decision Triggered")
            break
        end
    end
    @show vx vy t x0 PNR
    return tv, distance, power, power_abort, power_return, power_rendezvous, θv
end

function testFit()
end

function plotpower(t, power, power_abort, power_return, power_rendezvous)
    plot(t,power,label = "Available Power",lw=3)
    plot!(t,power_return,label = "Return Power",lw=3)
    plot!(t,power_abort,label = "Abort Power",lw=3)
    plot!(t,power_rendezvous,label = "Rendezvous Power",lw=3)
end

function runMission()
    t, distance, power, power_abort, power_return, power_rendezvous, θv = deterministicMission()
    plotpower(t, power, power_abort, power_return, power_rendezvous)
    plot!(t, distance,label = "Agent Distance",lw=3)
end

clearconsole()
#θdot = speed_profile(Array(0:1.0/100:1))
#plot(θdot)

t0 = 0
x0 = [500, 0]
L = [500, 0]
dt = 1.0
T = 200
vmax = 15.0
P = 2000

#animate_entities()
RDV, PNR, t, vx, vy = basicMPC()
@show RDV PNR t vx vy

RDVx_res = x0[1] + vx[1] * t[1] + vx[2] * t[2]
RDVy_res = x0[2] + vy[1] * t[1] + vy[2] * t[2]
RDV_t = t[1] + t[2]
RDV_θ = position_profile(RDV_t)
RDV_path = path(RDV_θ)

@show RDV RDVx_res RDVy_res RDV_path RDV_t RDV_θ

random_profile = false
