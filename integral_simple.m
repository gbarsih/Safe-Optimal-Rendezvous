
clc
clear
close all

r = 2;
mu = sym('mu_%d',[1 r+1]);
phi = sym('phi_%d',[1 r+1]);
syms theta_d theta theta_dot
syms t t1 t2
syms tt
syms theta_d
assume(theta_d>=0);
assume(theta_d<=1);
assume(t>=0);
%theta_dot = -(((t-100)/2000)^2) + 0.006;
theta_dot = 1/100 - 1/20000*t;
theta_dot = 1/200;
theta = int(theta_dot,t,0,t);
S = solve(theta==theta_d,t);
fun1 = matlabFunction(S(1));
%fun2 = matlabFunction(S(2));
% fun3 = matlabFunction(simplify(S(3)));

for i=0:r
    phi(i+1) = int(theta_dot^i,t,t1,t2); 
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

theta = matlabFunction(int(theta_dot,t,0,t));
%theta_dot = matlabFunction(theta_dot);
t = 0:0.1:200;

figure
clf
plot(t,theta(t))

figure(2)
theta = 0:0.01:1;
clf
plot(theta,fun1(theta))

%sanity check for integration

syms theta_h theta_d theta_dot_h theta_dot_d D t TR

theta_h = 0.1*t;
theta_d = 0.05*t;
theta_dot_h = 0.1;
theta_dot_d = 0.05;
D = theta_dot_h/-2;

theta_dot_d_hat = theta_dot_h + D

int(theta_dot_d_hat,t,0,TR)













