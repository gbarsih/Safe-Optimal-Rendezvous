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
    rem_power = rem_power - vx^2 * m / 2 * dt - vy^2 * m / 2 * dt
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
    1 / 200 .- (t ./ 2000 .- 1 / 20) .^ 2
end

function position_profile(t)
    -t^3 / 12000000 + t^2 / 40000 + t / 400 + 1 / 12
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
    θ_R = @NLexpression(MPC, -t_R^3 / 12000000 + t_R^2 / 40000 + t_R / 400 + 1 / 12)
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
    t_R = @NLexpression(MPC, t[2] + t[1] + t0)
    θ_R = @NLexpression(MPC, -t_R^3 / 12000000 + t_R^2 / 40000 + t_R / 400 + 1 / 12)
    @NLobjective(
        MPC,
        Min,
        1.0 * sum(vx[i]^2 * t[i] + vy[i]^2 * t[i] for i in [1 2 4]) +
        1.0 * (vx[3]^2 * t[3] + vy[3]^2 * t[3])
    )

    @constraint(MPC, PNR[1] == x[1] + vx[1] * t[1])
    @constraint(MPC, PNR[2] == x[2] + vy[1] * t[1])
    @constraint(MPC, RDV[1] == PNR[1] + vx[2] * t[2])
    @constraint(MPC, RDV[2] == PNR[2] + vy[2] * t[2])
    @constraint(MPC, L[1] == RDV[1] + vx[3] * t[3])
    @constraint(MPC, L[2] == RDV[2] + vy[3] * t[3])
    #@constraint(MPC, L[1] == PNR[1] + vx[4]*t[4])
    #@constraint(MPC, L[2] == PNR[2] + vy[4]*t[4])

    @NLconstraint(MPC, θ_R >= 0)
    @NLconstraint(MPC, θ_R <= 1)

    @NLconstraint(MPC, RDV[1] == θ_R * 1000)
    @NLconstraint(MPC, RDV[2] == θ_R * 1000)

    @constraint(MPC, t[1] + t[2] + t[3] <= T)
    @constraint(MPC, t[1] + t[2] + t[4] <= T)

    @NLconstraint(MPC, P >= sum((vx[i]^2 + vy[i]^2) * t[i] for i = 1:3))
    @NLconstraint(MPC, P >= sum((vx[i]^2 + vy[i]^2) * t[i] for i in [1 4]))

    optimize!(MPC)

    return value.(RDV), value.(PNR), value.(t), value.(vx), value.(vy)
end

function deterministicMission(
    x0 = [500.0, 0.0],
    L = [500.0, 0],
    P = 5000.0,
    T = 200.0,
    dt = 1.0,
)
    tv = Array(0:dt:T)
    N = length(tv)
    θd = 0
    for i = 1:N
        RDV, PNR, t, Vx, Vy = deterministicMPC(θd, tv[i], x0, L, dt, T, Inf, P)
        x = x0[1]
        y = x0[2]
        vx = Vx[1]
        vy = Vy[1]
        v = [vx vy]
        θR = position_profile(sum(t[1:2]))
        x0[1], x0[2], P = uav_dynamics(x, y, vx, vy, dt, P)
        θd = position_profile(tv[i]) #driver following prototypical profile
        @show tv[i] θd θR x0 RDV v P
        if norm(x0 .- path(θd)) < 5.0
            println("Rendezvous Successful")
            break
        end
    end
end

clearconsole()
θdot = speed_profile(Array(0:1.0/100:1))
plot(θdot)

t0 = 0
x0 = [500, 0]
L = [500, 0]
dt = 1 / 10
T = 200
vmax = 10.0
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

deterministicMission()
random_profile = false
