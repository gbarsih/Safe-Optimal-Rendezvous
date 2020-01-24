

using JuMP, Ipopt
using Plots, LinearAlgebra
default(dpi=100)
using BenchmarkTools
using Random
using StatsBase, StatsModels

clearconsole()

N = 15 #data points
x = collect(range(0,1,length=N)) #training points and data
t = sin.(2*pi.*x) + 0.2 .* randn(size(x))
X = collect(range(-0.5,1.5,length=100)) #test points and data
T = sin.(2*pi.*X)

a = 5e-3
b = 11.1
D = 8 #poly degree
M = D+1 #model param num
W = 0.5.*[1 -1]

s = zeros(size(X))
m = s

for j = 1:N



end


t2 = t
x2 = x
ϕ = broadcast(^, x2',0:D)
Sinv = a.*I + b.*(ϕ*ϕ')
Φ = broadcast(^, X', 0:D)
for k = 1:length(X)
    m[k] = b * Φ[:,k]'*(inv(Sinv)*ϕ*t2)
    s[k] = 1/b + Φ[:,k]'*(inv(Sinv)*Φ[:,k])
end #end for

scatter(x2,t2);
plot!(X,T)
plot!(X,m)
