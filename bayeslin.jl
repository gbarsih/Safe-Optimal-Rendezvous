# basis functions
linear(x::Number) = vcat(1, x)
linear(x::AbstractArray{T, 1}) where {T} = hcat(linear.(x)...)'

gaussian(x::Number, μ, σ=0.1) = vcat(1, exp.(-0.5*(x .- μ).^2/σ^2))
gaussian(x::AbstractArray{T, 1}, μ, σ=0.1) where {T} = hcat(gaussian.(x, Ref(μ), Ref(σ))...)'

polynomial(x::Number, r) = x.^r
polynomial(x::AbstractArray{T, 1}, r) where {T} = hcat(polynomial.(x, Ref(r))...)'

"""
    posterior(Y, ϕX, α, β)

Compute the posterior mean and variance given the observations Y,
regressor matrix ϕX, prior precision α, and the additive noise precision β.
"""
function posterior(Y, ϕX, α, β)
    Σ = inv(α*I + β*ϕX'*ϕX)
    μ = β*Σ*ϕX'*Y
    return μ, Σ
end

"""
    predict(ϕx, μ, Σ, β)

Compute the predictive distribution and standard deviation given the
regressed test point ϕx, posterior mean μ, posterior variance Σ, and
the additive noise precision β.
"""
function predict(ϕx, μ, Σ, β)
    μy = μ'*ϕx
    Σy = 1/β + ϕx'*Σ*ϕx
    return μy, √Σy
end

"""
    regress(Xo, Yo, Xt, Yt, ϕ, α, β)

Plot the point predictions given the observed data (Xo, Yo), test data (Xt, Yt),
basis functions ϕ, prior precision α, and the additive noise precision β.
"""
function regress(Xo, Yo, Xt, Yt, ϕ, α, β, ϕargs...)
    μ, Σ = posterior(Yo, ϕ(Xo, ϕargs...), α, β)

    # predictions
    Yμ = similar(Xt); Yσ = similar(Xt)
    for (i, xt) in enumerate(Xt)
        yμ, yσ = predict(ϕ(xt, ϕargs...), μ, Σ, β)
        Yμ[i] = yμ; Yσ[i] = yσ
    end

    Prediction1D(Xo, Yo, Xt, Yt, Yμ, Yσ)
end

struct Prediction1D{T}
    Xo::T
    Yo::T
    Xt::T
    Yt::T
    μ::T
    σ::T
end

@recipe function f(p::Prediction1D)
    legend --> false
    @series begin
        fillcolor --> :blue
        fillalpha --> 0.2
        linewidth --> 1.0
        linecolor --> :blue
        ribbon := p.σ
        (p.Xt, p.μ)
    end
    @series begin
        linewidth --> 2.0
        linecolor --> :black
        linestyle --> :dash
        (p.Xt, p.Yt)
    end
    @series begin
        seriestype --> :scatter
        markercolor --> :red
        markershape --> :circle
        markersize --> 3.0
        markerstrokecolor --> :red
        (p.Xo, p.Yo)
    end
end

function linefit(N, α=2.0, β=25.0; seeded=false)
    f(x, β=Inf) = -0.3 + 0.5*x + 1/β*randn()
    if seeded
        seed!(1729)
    end
    Xo = 2.0*rand(N) .- 1.0
    Yo = f.(Xo, β)
    Xt = collect(-1.0:0.01:1.0)
    Yt = f.(Xt)
    regress(Xo, Yo, Xt, Yt, linear, α, β)
end

function gaussfit(N, α=1.0, β=25.0, μ=0:0.1:3; seeded=false)
    f(x, β=Inf) = 0.5 + x*sin(2*π*x) + 1/β*randn()
    if seeded
        seed!(1729)
    end
    Xo = rand(N)*3.0
    Yo = f.(Xo, β)
    Xt = collect(0.0:0.01:3.0)
    Yt = f.(Xt)
    regress(Xo, Yo, Xt, Yt, gaussian, α, β, μ)
end

function polyfit(N, α=0.005, β=1/(0.3^2), r=0:5; seeded=false)
    f(x, β=Inf) = 0.5 + sin(2*π*x) + 1/β*randn()
    if seeded
        seed!(1729)
    end
    Xo = rand(N)
    Yo = f.(Xo, β)
    Xt = collect(0.0:0.005:1.0)
    Yt = f.(Xt)
    regress(Xo, Yo, Xt, Yt, polynomial, α, β, r)
end
