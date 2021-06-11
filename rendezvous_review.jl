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
default(dpi = 200)
default(size = [1200, 800])
#theme(:juno)
theme(:default)
mo = [1.0 1.0 0.5 1.0]

function uav_dynamics(x, y, vx, vy, dt, rem_power = Inf, vmax = Inf, m = 1.0)
    vx > vmax ? vmax : vx
    vy > vmax ? vmax : vy
    vx < -vmax ? -vmax : vx
    vy < -vmax ? -vmax : vy
    x = x + vx * dt
    y = y + vy * dt
    rem_power = rem_power - vx^2 * m * dt - vy^2 * m * dt - 1 * dt
    return x, y, rem_power
end

function driver_dynamics(θ, t, dt)
    return θ .+ speed_profile(t) .* dt
end

function path(θ)
    x = θ .* 1.0
    y = θ .* 1.0
    return x, y
end

function plot_path(θ0, n = 100, bg = "white", cgr = false)
    θ = Array(θ0:1.0/n:1000)
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
    5.0
    10 .* (1 .- t ./ 200)
end

function position_profile(t)
    #-t^3 / 12000000 + t^2 / 40000 + t / 400 + 1 / 12
    #-(t.*(t .- 400))./40000
    t .* 5.0
    .-(t .* (t .- 400)) ./ 40
end

function time_profile(θd)
    #200 - 20000*(1/10000 - θd/10000)^(1/2)
    200 - 20 * (100 - θd / 10)^(1 / 2)
end

function animate_entities()
    n = 10
    t = Array(0:1/n:200)
    N = length(t)
    D = sqrt(2)
    xdriver, ydriver = path(0)
    θ = 0
    x = 0
    y = 500
    p = 2000
    anim = @animate for i = 1:N
        plot(label = false)
        plot_path(0)
        scatter!(
            path(θ),
            markersize = 10.0,
            markershape = :square,
            color = :green,
        )
        scatter!(
            [x],
            [y],
            markersize = 10.0,
            markershape = :circle,
            color = :black,
        )
        s1 = @sprintf(
            "Driver's speed = %15.2f km/h",
            speed_profile(t[i]) * D * 3.6
        )
        annotate!(0, 1000, text(string(s1), 20, :left))
        θ = driver_dynamics(θ, t[i], 1 / n)
        θi = position_profile(t[i])
        x, y, p = uav_dynamics(x, y, 2.5, 0, 1 / n, p)
        @show θ θi x y p
    end every 50
    gif(anim, "driver_anim.gif", fps = 10)
end

function deterministicMPC(
    θd,
    PNRp,
    vxp,
    vyp,
    θRp,
    tp,
    t0 = 0.0,
    x = [500.0, 0.0],
    L = [500.0, 0.0],
    dt = 1.0,
    T = 200.0,
    vmax = Inf,
    P = 5000.0,
    random_profile = false,
)
    MPC =
        Model(with_optimizer(Ipopt.Optimizer, print_level = 0, max_iter = 500))
    @variable(MPC, PNR[i = 1:2])
    @variable(MPC, RDV[i = 1:2])
    @variable(MPC, -vmax <= vx[i = 1:4] <= vmax) #controls
    @variable(MPC, -vmax <= vy[i = 1:4] <= vmax) #controls
    @variable(MPC, 0.1 <= t[i = 1:4]) #controls
    t_i = time_profile(θd) #relative adjusted time
    t_R = @NLexpression(MPC, t[2] + t[1] + t0) #absolute rendezvous time
    t_a = @NLexpression(MPC, t_R - t0 + t_i) #adjusted rendezvous time
    θ_R = @NLexpression(MPC, -(t_a * (t_a - 400)) / 40) #rendezvous location
    #    @NLobjective(
    #        MPC,
    #        Min,
    #        1.0 * sum(vx[i]^2 * t[i] + vy[i]^2 * t[i] for i in [1 2 4]) +
    #        1.0 * (vx[3]^2 * t[3] + vy[3]^2 * t[3])
    #    )
    @objective(MPC, Min, sum(t[i] for i = 1:4))
    r = 10
    srv = 1.0
    srt = 0.3
    tr = 0.3
    #@constraint(MPC, (PNRp[1] - PNR[1])^2 <= r^2)
    #@constraint(MPC, (PNRp[2] - PNR[2])^2 <= r^2)
    for i = 1:4
        #@constraint(MPC, (vxp[i] - vx[i])^2 <= srv^2)
        #@constraint(MPC, (vyp[i] - vy[i])^2 <= srv^2)
        #@constraint(MPC, (t[i] - t[i])^2 <= srt^2)
    end
    #@NLconstraint(MPC, (θRp - θ_R)^2 <= tr^2)
    @constraint(MPC, PNR[1] == x[1] + vx[1] * t[1])
    @constraint(MPC, PNR[2] == x[2] + vy[1] * t[1])
    @constraint(MPC, RDV[1] == PNR[1] + vx[2] * t[2])
    @constraint(MPC, RDV[2] == PNR[2] + vy[2] * t[2])
    @constraint(MPC, L[1] == RDV[1] + vx[3] * t[3])
    @constraint(MPC, L[2] == RDV[2] + vy[3] * t[3])
    @constraint(MPC, L[1] == PNR[1] + vx[4] * t[4])
    @constraint(MPC, L[2] == PNR[2] + vy[4] * t[4])
    @NLconstraint(MPC, θ_R >= 0)
    @NLconstraint(MPC, θ_R <= 1000)
    @NLconstraint(MPC, RDV[1] == θ_R)
    @NLconstraint(MPC, RDV[2] == θ_R)
    @constraint(MPC, t[1] + t[2] + t[3] <= T)
    @constraint(MPC, t[1] + t[2] + t[4] <= T)
    @NLconstraint(
        MPC,
        P - 100 >=
        sum((vx[i]^2 + vy[i]^2) * t[i] * mo[i] + 1 * t[i] for i = 1:3)
    )
    @NLconstraint(
        MPC,
        P - 100 >=
        sum((vx[i]^2 + vy[i]^2) * t[i] * mo[i] + 1 * t[i] for i in [1 4])
    )
    optimize!(MPC)
    return value.(RDV), value.(PNR), value.(t), value.(vx), value.(vy)
end

function deterministicMission(
    x0 = [500.0, 0.0],
    L = [500.0, 0],
    P = 8000.0,
    T = 150.0,
    vmax = 10.0,
    dt = 1.0,
    ns = 10,
)
    tv = Array(0:dt:T)
    N = length(tv)
    distance = zeros(N)
    power = zeros(N)
    power_abort = zeros(N)
    power_return = zeros(N)
    power_rendezvous = zeros(N)
    θv = zeros(N)
    μv = zeros(N)
    θd = 0.0
    θR = 500
    PNR = x0
    Vx = [0.0, 0.0, 0.0, 0.0]
    Vy = [0.0, 0.0, 0.0, 0.0]
    t = [10.0, 10.0, 10.0, 10.0]
    Tm = 0
    β = 1 / (0.3^2)
    α = 0.005
    tf = Array(0:1:(ns-1)) #seed GP
    Xo = zeros(N + ns)
    Yo = zeros(N + ns)
    Xo[1:ns] = speed_profile(tf)
    Yo[1:ns] = D.(Xo[1:ns], β) #observed deviation
    μ, Σ = posterior(Yo, linear(Xo), α, β)
    phi(t, μ) = t * μ[1] - ((t * (t - 400)) / 40) * μ[2]
    for i = 1:N
        RDV, PNR, t, Vx, Vy = deterministicMPC(
            θd,
            PNR,
            Vx,
            Vy,
            θR,
            t,
            tv[i],
            x0,
            L,
            dt,
            T,
            vmax,
            P,
        )
        x = x0[1]
        y = x0[2]
        vx = Vx[1]
        vy = Vy[1]
        v = [vx vy]
        θR = position_profile(sum(t[1:2]) + tv[i])
        x0[1], x0[2], P = uav_dynamics(x, y, vx, vy, dt, P)
        θd = position_profile(1.0 * tv[i]) + 1 / β * randn()#driver following prototypical profile
        Xo[ns+i] = speed_profile(tv[i])
        Yo[ns+i] = D(Xo[ns+i], β)
        μ, Σ = posterior(Yo[1:(ns+i)], linear(Xo[1:(ns+i)]), α, β)
        μv[i] = μ[2]
        Pa =
            t[1] * (Vx[1]^2 + Vy[1]^2) * mo[1] +
            t[4] * (Vx[4]^2 + Vy[4]^2) * mo[4]
        Pr = t[3] * (Vx[3]^2 + Vy[3]^2) * mo[3]
        Pd =
            t[1] * (Vx[1]^2 + Vy[1]^2) * mo[1] +
            t[2] * (Vx[2]^2 + Vy[2]^2) * mo[2]
        distance[i] = norm(x0 .- path(θd))
        power[i] = P
        power_abort[i] = Pa
        power_return[i] = Pr
        power_rendezvous[i] = Pd
        θv[i] = θR
        cRDVx = Vx[1] * t[1] + Vx[2] * t[2] + x0[1]
        cRDVy = Vy[1] * t[1] + Vy[2] * t[2] + x0[2]
        cLx = Vx[1] * t[1] + Vx[2] * t[2] + Vx[3] * t[3] + x0[1]
        cLy = Vy[1] * t[1] + Vy[2] * t[2] + Vy[3] * t[3] + x0[2]
        dist = norm(x0 .- path(θd))
        Tm = i
        @show tv[i] θd θR x0 RDV dist v P Pa Pr Pd cRDVx cRDVy cLx cLy
        if norm(x0 .- path(θd)) < 5.0
            println("Rendezvous Successful")
            break
        elseif (Pa > P + 100 || Pr > P + 100)
            println("Abort Decision Triggered")
            break
        end
    end
    return tv,
    distance,
    power,
    power_abort,
    power_return,
    power_rendezvous,
    θv,
    Tm,
    μv
end

function riskMPC(
    θd,
    PNRp,
    vxp,
    vyp,
    θRp,
    tp,
    μ,
    Σ,
    t0 = 0.0,
    x = [500.0, 0.0],
    L = [500.0, 0.0],
    dt = 1.0,
    T = 200.0,
    vmax = Inf,
    P = 5000.0,
    random_profile = false,
)
    MPC =
        Model(with_optimizer(Ipopt.Optimizer, print_level = 0, max_iter = 500))
    @variable(MPC, PNR[i = 1:2])
    @variable(MPC, RDV[i = 1:2])
    @variable(MPC, -vmax <= vx[i = 1:4] <= vmax) #controls
    @variable(MPC, -vmax <= vy[i = 1:4] <= vmax) #controls
    @variable(MPC, 0.1 <= t[i = 1:4]) #controls
    t_i = time_profile(θd) #relative adjusted time
    t_R = @NLexpression(MPC, t[2] + t[1] + t0) #absolute rendezvous time
    t_a = @NLexpression(MPC, t_R - t0 + t_i) #adjusted rendezvous time
    #θ_R = @NLexpression(MPC, -(t_a * (t_a - 400)) / 40) #rendezvous location
    P1 = @NLexpression(MPC, t_R)
    P2 = @NLexpression(MPC, -(t_R * (t_R - 400)) / 40)
    θ_R = @NLexpression(MPC, μ[1] * P1 + μ[2] * P2)
    Σ_R = @NLexpression(
        MPC,
        P1 * (P1 * Σ[1, 1] + P2 * Σ[2, 1]) + P2 * (P1 * Σ[1, 2] + P2 * Σ[2, 2])
    )

    #(t[2] + t[1] + t0)
    #(-((t[2] + t[1] + t0) * ((t[2] + t[1] + t0) - 400)) / 40))
    #@NLobjective(
    #    MPC,
    #    Min,
    #    mo[1] * sum(vx[i]^2 * t[i] + vy[i]^2 * t[i] for i in [1 2 4]) +
    #    mo[2] * (vx[3]^2 * t[3] + vy[3]^2 * t[3] - 1000 * t[1])
    #)
    @NLobjective(MPC, Min, sum(t[i] for i = 1:4) - 1000 * t[1] + 100 * (Σ_R))
    r = 10
    srv = 1.0
    srt = 0.3
    tr = 0.3
    #@constraint(MPC, (PNRp[1] - PNR[1])^2 <= r^2)
    #@constraint(MPC, (PNRp[2] - PNR[2])^2 <= r^2)
    for i = 1:4
        #@constraint(MPC, (vxp[i] - vx[i])^2 <= srv^2)
        #@constraint(MPC, (vyp[i] - vy[i])^2 <= srv^2)
        #@constraint(MPC, (t[i] - t[i])^2 <= srt^2)
    end
    #@NLconstraint(MPC, (θRp - θ_R)^2 <= tr^2)
    @constraint(MPC, PNR[1] == x[1] + vx[1] * t[1])
    @constraint(MPC, PNR[2] == x[2] + vy[1] * t[1])
    @constraint(MPC, RDV[1] == PNR[1] + vx[2] * t[2])
    @constraint(MPC, RDV[2] == PNR[2] + vy[2] * t[2])
    @constraint(MPC, L[1] == RDV[1] + vx[3] * t[3])
    @constraint(MPC, L[2] == RDV[2] + vy[3] * t[3])
    @constraint(MPC, L[1] == PNR[1] + vx[4] * t[4])
    @constraint(MPC, L[2] == PNR[2] + vy[4] * t[4])
    #@NLconstraint(MPC, θ_R >= 0)
    #@NLconstraint(MPC, θ_R <= 1000)
    @NLconstraint(MPC, RDV[1] == θ_R)
    @NLconstraint(MPC, RDV[2] == θ_R)
    @constraint(MPC, t[1] + t[2] + t[3] <= T)
    @constraint(MPC, t[1] + t[2] + t[4] <= T)
    @NLconstraint(
        MPC,
        P - 100 >=
        sum((vx[i]^2 + vy[i]^2) * t[i] * mo[i] + 1 * t[i] for i = 1:3)
    )
    @NLconstraint(
        MPC,
        P - 100 >=
        sum((vx[i]^2 + vy[i]^2) * t[i] * mo[i] + 1 * t[i] for i in [1 4])
    )
    optimize!(MPC)
    return value.(RDV), value.(PNR), value.(t), value.(vx), value.(vy)
end

function riskMission(
    x0 = [500.0, 0.0],
    L = [500.0, 0],
    P = 8000.0,
    T = 200.0,
    vmax = 10.0,
    dt = 1.0,
    ns = 30,
)
    tv = Array(0:dt:T)
    N = length(tv)
    distance = zeros(N)
    power = zeros(N)
    power_abort = zeros(N)
    power_return = zeros(N)
    power_rendezvous = zeros(N)
    θv = zeros(N)
    μv = zeros(N)
    t1 = zeros(N)
    DSR = zeros(N)
    θd = 0.0
    θR = 500
    PNR = x0
    Vx = [0.0, 0.0, 0.0, 0.0]
    Vy = [0.0, 0.0, 0.0, 0.0]
    t = [10.0, 10.0, 10.0, 10.0]
    Tm = 0
    β = 1 / (6^2)
    α = 0.005
    tf = Array(0:1:(ns-1)) #seed GP
    Xo = zeros(N + ns)
    Yo = zeros(N + ns)
    Xo[1:ns] = speed_profile(tf)
    Yo[1:ns] = D.(Xo[1:ns], β) #observed deviation
    μ, Σ = posterior(Yo, linear(Xo), α, β)
    phi(t, μ) = t * μ[1] - ((t * (t - 400)) / 40) * μ[2]
    for i = 1:N
        RDV, PNR, t, Vx, Vy =
            riskMPC(θd, PNR, Vx, Vy, θR, t, μ, Σ, tv[i], x0, L, dt, T, vmax, P)
        x = x0[1]
        y = x0[2]
        vx = Vx[1]
        vy = Vy[1]
        v = [vx vy]
        θR = position_profile(sum(t[1:2]) + tv[i])
        x0[1], x0[2], P = uav_dynamics(x, y, vx, vy, dt, P)
        #θd = position_profile(1.00 * tv[i]) #driver following prototypical profile
        #θd = θd + speed_profile(tv[i])*dt
        θd = θd + D(speed_profile(tv[i])) * dt
        Xo[ns+i] = speed_profile(tv[i])
        Yo[ns+i] = D(Xo[ns+i], β)
        μ, Σ = posterior(Yo[1:(ns+i)], linear(Xo[1:(ns+i)]), α, β)
        μv[i] = μ[2]

        Pa =
            t[1] * (Vx[1]^2 + Vy[1]^2) * mo[1] +
            t[4] * (Vx[4]^2 + Vy[4]^2) * mo[4]
        Pr = t[3] * (Vx[3]^2 + Vy[3]^2) * mo[3]
        Pd =
            t[1] * (Vx[1]^2 + Vy[1]^2) * mo[1] +
            t[2] * (Vx[2]^2 + Vy[2]^2) * mo[2]
        distance[i] = norm(x0 .- path(θd))
        power[i] = P
        power_abort[i] = Pa
        power_return[i] = Pr
        power_rendezvous[i] = Pd
        θv[i] = θR
        t1[i] = t[1]
        cRDVx = Vx[1] * t[1] + Vx[2] * t[2] + x0[1]
        cRDVy = Vy[1] * t[1] + Vy[2] * t[2] + x0[2]
        cLx = Vx[1] * t[1] + Vx[2] * t[2] + Vx[3] * t[3] + x0[1]
        cLy = Vy[1] * t[1] + Vy[2] * t[2] + Vy[3] * t[3] + x0[2]
        dist = norm(x0 .- path(θd))
        Tm = i
        DSRisk = DownsideRisk(x0, Σ, t, Vx, Vy, θR, L)
        DSR[i] = DSRisk
        #@show tv[i] θd θR x0 RDV dist v P Pa Pr Pd cRDVx cRDVy cLx cLy
        @show DSRisk, tv[i]
        if norm(x0 .- path(θd)) < 10.0
            println("Rendezvous Successful")
            break
        elseif (Pa > P + 100 || Pr > P + 100)
            println("Abort Decision Triggered")
            break
        elseif (t[1] < 5.0)
            println("Risk-Based Decision Triggered")
            break
        end
    end
    DSR[1] = DSR[2]
    return tv,
    distance,
    power,
    power_abort,
    power_return,
    power_rendezvous,
    θv,
    Tm,
    μv,
    t1,
    DSR
end

function D(v, β = Inf)
    1.3 * v + 1 / β * randn()
end

function DownsideRisk(x, Σ, t, vx, vy, θR, L)
    t_R = t[1] + t[2]
    P1 = t_R
    P2 = -(t_R * (t_R - 400)) / 40
    θv = P1 * (P1 * Σ[1, 1] + P2 * Σ[2, 1]) + P2 * (P1 * Σ[1, 2] + P2 * Σ[2, 2])
    EnergyRisk(θR, θv, x, L, t, vx, vy)
end

function EnergyRisk(θR, θv, x, L, t, vx, vy)
    xR = path(θR)
    xvplus = path(θR + θv)
    xvminus = path(θR - θv)
    OldDistance = [EuclideanDistance(x, xR) EuclideanDistance(xR, L)]
    PlusDistance = [EuclideanDistance(x, xvplus) EuclideanDistance(xvplus, L)]
    MinusDistance =
        [EuclideanDistance(x, xvminus) EuclideanDistance(xvminus, L)]
    if sum(PlusDistance) > sum(MinusDistance)
        direction = 1.0
        NewDistance = PlusDistance
    else
        direction = -1.0
        NewDistance = MinusDistance
    end
    #Compute Energy Risk
    vx_new = copy(vx)
    vy_new = copy(vy)
    for i = 1:2
        gain = NewDistance[i] / OldDistance[i]
        vx_new[i+1] = vx[i+1] * gain
        vy_new[i+1] = vy[i+1] * gain
    end

    OldEnergy = RendezvousEnergy(vx, vy, t)
    NewEnergy = RendezvousEnergy(vx_new, vy_new, t)
    abs(NewEnergy - OldEnergy)

end

function RendezvousEnergy(vx, vy, t)
    sum((vx[i]^2 + vy[i]^2) * t[i] * mo[i] + 1 * t[i] for i in [2 3])
end

function EuclideanDistance(x1, x2)
    sqrt((x1[1] - x2[1])^2 + (x1[2] - x2[2])^2)
end

function testFit(N = 10, α = 0.005, β = 1 / (0.6^2), r = 0:1)
    default(dpi = 300)
    #default(thickness_scaling = 2)
    #default(size = [1200, 800])
    t = Array(0:1:N)
    Xo = speed_profile(t)
    #Xo = 0.0 .+ rand(N) ./ 1 #random samples
    #Xo = 0.2 .+ 0.1 * rand(N) #random samples
    Yo = D.(Xo, β) #observed deviation
    t = Array(0:1:200)
    Xt = speed_profile(t)
    Yt = D.(Xt, Inf) #actual deviation
    s = @sprintf "Fit Performance: %ds" N
    plot(
        regress(Xo, Yo, Xt, Yt, polynomial, α, β, r),
        xlabel = "Prototypical Speed [θ/s]",
        ylabel = "Driver's Speed [θ/s]",
        title = s,
    )
end

function fitWeights(N = 10, α = 0.005, β = 1 / (0.6^2), r = 0:2)
    t = Array(0:1:N)
    Xo = speed_profile(t)
    #Xo = 0.0 .+ rand(N) ./ 1 #random samples
    #Xo = 0.2 .+ 0.1 * rand(N) #random samples
    Yo = D.(Xo, β) #observed deviation
    Xt = collect(0:1:50.0)
    Yt = D.(Xt, Inf) #actual deviation
    μ, Σ = posterior(Yo, linear(Xo), α, β)
    return μ, Σ
end

function plotpower(
    t,
    power,
    power_abort,
    power_return,
    power_rendezvous,
    Tm = 0.0,
)
    plot(t, power, label = "Available Energy", lw = 3)
    plot!(t, power_return, label = "Return Energy", lw = 3)
    plot!(t, power_abort, label = "Abort Energy", lw = 3)
    plot!(t, power_rendezvous, label = "Rendezvous Energy", lw = 3)
    plot!(
        xlabel = "Time [s]",
        ylabel = "Energy [J]",
        title = "Energy Profile",
        xlims = (0, t[Tm]),
        ylims = (0, 10e3),
    )

end

function runDeterministcMission()
    default(dpi = 300)
    default(thickness_scaling = 2)
    default(size = [1200, 800])
    t,
    distance,
    power,
    power_abort,
    power_return,
    power_rendezvous,
    θv,
    Tm,
    μv = deterministicMission()
    l = @layout [a b]
    p1 = plotpower(t, power, power_abort, power_return, power_rendezvous, Tm)
    p2 = plot(t, distance, label = "Agent Distance", lw = 3, xlims = (0, t[Tm]))
    p2 = plot!(
        xlabel = "Time [s]",
        ylabel = "Distante [m]",
        title = "UAV-Driver Distance",
    )
    plot(p1, p2, layout = l)
    #plot!(t, μv .* 1000, label = "Mu", lw = 3, xlims = (0, t[Tm])
end

function runMission(s = nothing)
    default(dpi = 300)
    #default(thickness_scaling = 2)
    #default(size = [1200, 800])
    if s != nothing
        seed!(s)
    end
    t,
    distance,
    power,
    power_abort,
    power_return,
    power_rendezvous,
    θv,
    Tm,
    μv,
    t1,
    DSR = riskMission()
    @show t1
    if Tm > 5
        Tm = Tm - 3
    end
    l = @layout [a b]
    p1 = plotpower(t, power, power_abort, power_return, power_rendezvous, Tm)
    p1 = plot!(t, DSR, label = "Downside Potential", lw = 3, xlims = (0, t[Tm]))
    p1 = hline!(
        [300],
        label = "Downside Potential Threshold",
        lw = 3,
        xlims = (0, t[Tm]),
    )
    p2 = plot(t, distance, label = "Agent Distance", lw = 3, xlims = (0, t[Tm]))
    p2 = plot!(
        xlabel = "Time [s]",
        ylabel = "Distance [m]",
        title = "UAV-Driver Distance",
    )
    p = plot(p1, p2, layout = l)
    display(p)
    s = "/Users/gabrielbarsi/Documents/GitHub/Safe-Optimal-Rendezvous/Safe-Optimal-Rendezvous/energy.png"
    savefig(s)
end

function benchmarkDeterministic()
    θd = 0.0
    PNRp = [0.0 0.0]
    vxp = 0.0
    vyp = 0.0
    θRp = 500
    tp = 0.0
    @benchmark deterministicMPC($θd, $PNRp, $vxp, $vyp, $θRp, $tp)
end

function benchmarkRisk(
    x0 = [500.0, 0.0],
    L = [500.0, 0],
    P = 80000.0,
    T = 200.0,
    vmax = 15.0,
    dt = 1.0,
    ns = 30,
)
    tv = Array(0:dt:T)
    N = length(tv)
    distance = zeros(N)
    power = zeros(N)
    power_abort = zeros(N)
    power_return = zeros(N)
    power_rendezvous = zeros(N)
    θv = zeros(N)
    μv = zeros(N)
    t1 = zeros(N)
    DSR = zeros(N)
    θd = 0.0
    θR = 500
    PNR = x0
    Vx = [0.0, 0.0, 0.0, 0.0]
    Vy = [0.0, 0.0, 0.0, 0.0]
    t = [10.0, 10.0, 10.0, 10.0]
    Tm = 0
    β = 1 / (0.3^2)
    α = 0.005
    tf = Array(0:1:(ns-1)) #seed GP
    Xo = zeros(N + ns)
    Yo = zeros(N + ns)
    Xo[1:ns] = speed_profile(tf)
    Yo[1:ns] = D.(Xo[1:ns], β) #observed deviation
    μ, Σ = posterior(Yo, linear(Xo), α, β)
    phi(t, μ) = t * μ[1] - ((t * (t - 400)) / 40) * μ[2]
    @benchmark riskMPC(
        $θd,
        $PNR,
        $Vx,
        $Vy,
        $θR,
        $t,
        $μ,
        $Σ,
        $tv[1],
        $x0,
        $L,
        $dt,
        $T,
        $vmax,
        $P,
    )
end

function plotFittingPerf(n = 10, m = 50)
    seed!(0)
    p1 = testFit(n)
    seed!(0)
    p2 = testFit(m)
    l = @layout [a b]
    p = plot(p1, p2, layout = l, ylims = (-5, 15))
    display(p)
    #s = "/Users/gabrielbarsi/Documents/GitHub/Safe-Optimal-Rendezvous/Safe-Optimal-Rendezvous/fitperf.png"
    #savefig(s)


    μl, Σl = fitWeights(n)
    θvl(t) =
        t .* (t .* Σl[1, 1] .+ .-(t .* (t .- 400)) ./ 40 * Σl[2, 1]) .+
        .-(t .* (t .- 400)) ./ 40 .*
        (t .* Σl[1, 2] .+ .-(t .* (t .- 400)) ./ 40 .* Σl[2, 2])
    θl(t) = μl[1] .* t + μl[2] .* .-(t .* (t .- 400)) ./ 40

    μh, Σh = fitWeights(m)
    θvh(t) =
        t .* (t .* Σh[1, 1] + .-(t .* (t .- 400)) ./ 40 * Σh[2, 1]) .+
        .-(t .* (t .- 400)) ./ 40 .*
        (t .* Σh[1, 2] .+ .-(t .* (t .- 400)) ./ 40 .* Σh[2, 2])
    θh(t) = μh[1] .* t .+ μh[2] .* .-(t .* (t .- 400)) ./ 40

    #t = Array(0:1:200)
    #s1 = @sprintf("%ds",n)
    #s2 = @sprintf("%ds",m)
    #plot(t, θl(t), ribbon=θvl(t), label=s1)
    #p3 = plot!(t, θh(t), ribbon=θvh(t), label=s2, xlabel="Time [s]", ylabel="")
    #l2 = @layout [a b ; c]
    #pf = plot(p1, p2, p3, layout = l2)

end
