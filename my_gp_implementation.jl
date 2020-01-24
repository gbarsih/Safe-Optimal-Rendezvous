using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using JuMP

function kernelf(X1, X2, l=1.0, sigma_f=1.0)

    # Isotropic squared exponential kernel. Computes
    # a covariance matrix from points in X1 and X2.

    #Σ = zeros(size(X1,1), size(X2,1))
    Σ = Array{Float64,2}(undef, size(X1,1), size(X2,1))
    for i in 1:size(Σ,1)
        for j in 1:size(Σ,2)
            Σ[i,j] = sigma_f^2*exp(-(X1[i]-X2[j])^2/(2*l^2))
        end
    end

    return Σ

end

function plot_gp(μ, Σ, X, X_train=nothing, Y_train=nothing)
    σ = 1.96*sqrt.(diag(Σ))
    p = plot([X X], [μ μ], fillrange=[μ+σ μ-σ], fillalpha=0.3, c=:blue, label="")
    plot!(X, μ, c=:blue, label="GP mean")
    if X_train!=nothing && Y_train!=nothing
        plot!(X_train, Y_train, seriestype="scatter", color="red", label="samples")
    end
    display(p)
end

function kernelTest()
    X1 = collect(range(-5, step=1, 5))
    X2 = collect(range(-5, step=1, 5)) .+ 0.1.*randn(size(X1))
    plot(X1, label="original data")
    plot!(X2, label="corrupted data")
    μ = zeros(size(X1))
    Σ = kernelf(X1, X1)
    plot_gp(μ, Σ, X1)
end

function posteriorPredictor(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, K_inv=nothing)
    K_s = kernelf(X_train, X_s, l, sigma_f)
    K_ss = kernelf(X_s, X_s, l, sigma_f) + 1e-8*I
    if K_inv == nothing
        K = kernelf(X_train, X_train, l, sigma_f) + sigma_y^2*I
        K_inv = inv(K)
    end

    μ_s = K_s'*K_inv*Y_train
    #mu_s = K_s.T.dot(K_inv).dot(Y_train)
    Σ_s = K_ss - K_s'*K_inv*K_s
    #cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return μ_s, Σ_s, K_inv
end

function testPredictor()
    lb = 0;
    ub = 5;
    n = 5;
    sigma_y = 0.0
    X_train = [-4 -3 -2 -1 1 4]'
    X_train = rand(n) .*(ub-lb) .+lb
    Y_train = sin.(X_train) + sigma_y*randn(size(X_train,1))
    X = collect(range(lb, ub, length=n))
    μ_s, Σ_s = posteriorPredictor(X, X_train, Y_train, 1.0, 1.0, sigma_y)
    plot_gp(μ_s, Σ_s, X, X_train, Y_train)
    plot!(X, sin.(X), lw=:2, color=:black, label="true function")
end

function hyperParameterOptimization(X_train, Y_train, noise, naive=true)
    model = Model()
    @variable(theta[1:2] >= 1e-5) #hyperparam variables
    if naive
    end
end


function f(x)
    return 8*sin.(x) + 1/2*x.^2 + 1/8*x.^3 + 1/32*x.^4 - 1/128*x.^5
end

#kernelTest()
#testPredictor()

lb = 1;
ub = 7;
n = 100;
l = 1.0;
sigma_f = 1.0
sigma_y = 0.9
#X_train = [-4 -3 -2 -1 1 4]'
#X_train = rand(n).*(ub-lb) .+ lb
X_train = 1.1*randn(n) .+ (ub - lb)/2 .+ lb
Y_train = f(X_train) + sigma_y*randn(size(X_train,1))
X = collect(range(lb, ub, length=n))
if sigma_y != 0.0
    μ_s, Σ_s, K_inv = posteriorPredictor(X, X_train, Y_train, l, sigma_f, sigma_y)
else
    μ_s, Σ_s, K_inv = posteriorPredictor(X, X_train, Y_train, l, sigma_f)
end
println("Timing training + inference")
@time posteriorPredictor(X, X_train, Y_train, 1.0, 1.0, sigma_y)
println("Timing inference")
@time posteriorPredictor(X, X_train, Y_train, 1.0, 1.0, sigma_y, K_inv)
plot_gp(μ_s, Σ_s, X, X_train, Y_train)
plot!(X, f(X), lw=:2, color=:black, label="true function", xlim=(lb,ub))
