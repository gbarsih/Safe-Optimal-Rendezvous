
clc
clear

r = 2;
mu = sym('mu_%d',[1 r+1]);
phi = sym('phi_%d',[1 r+1]);
syms theta_d
syms t t1 t2
syms tt

theta = (0.01 + 0.01*cos(4*pi*t/10));

for i=0:r
    phi(i+1) = int(theta^i,t,t1,t2);
end

theta_d = 0;
for i=0:r
    theta_d = mu(i+1)*phi(i+1) + theta_d;
end
    
phi_s = char(phi);
phi_s([1:8, end-1:end]) = []
simplify(theta_d)
%simplify(sin(5+4.5*sin(2*pi*(tt + theta_d))))
simplify((tt + theta_d))

Sigma = sym('S', [3 3])
Psi = sym('P',[3 1])

simplify(Psi.'*Sigma*Psi)