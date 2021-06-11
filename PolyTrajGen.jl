"""

 █████╗  ██████╗██████╗ ██╗
██╔══██╗██╔════╝██╔══██╗██║
███████║██║     ██████╔╝██║
██╔══██║██║     ██╔══██╗██║
██║  ██║╚██████╗██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

File:       PlanarQuadMPC.jl
Author:     Gabriel Barsi Haberfeld, 2020. gbh2@illinois.edu
Function:   This program implemetns an MPC controller for trajectory tracking of
            a planar quadrotor.

Instructions:   Run this file in juno with Julia 1.5.1 or later.
Requirements:   JuMP, Ipopt, Plots, LinearAlgebra, BenchmarkTools.

"""

using JuMP, Ipopt
using Plots, LinearAlgebra
using Polynomials
using SparseArrays
using ControlSystems

function computeTraj()
    m_q = 4                                #mass of a quadrotor
    I_q = diag([0.3 0.3 0.3])              #moment of inertia of a quadrotor
    g = 9.81                               #gravitational acceleration

    #Trajectory
    global m
    n = 4                                  #number of flat outputs (x, y, z, psi)
    t_f = 10                               #final time of the trajectory

    order = 6                              #order of polynomial functions

    time_interval_selection_flag = true    #true : fixed time interval, false : optimal time interval
    if (time_interval_selection_flag)
        t = collect(range(0, stop = t_f, length = m + 1))
    end

    n_intermediate = 5
    corridor_width = 0.05
    corridor_position = [3 4]

    #keyframes, column = index, rows = xyz yaw
    keyframe = [
        0 7.1 14.3 7.9
        0 8.6 0.7 -5.3
        0 1.0 1.3 0.9
        0 0 0 0
    ]
    m = size(keyframe,2)
    c = zeros(4 * (order + 1) * m)
    mu_r = 1
    mu_psi = 1
    k_r = 4
    k_psi = 2
    A = computeCostMat(order, m, mu_r, mu_psi, k_r, k_psi, t)
end

function computeCostMat(order, m, mu_r, mu_psi, k_r, k_psi, t)
    polynomial_r = Polynomial(ones(order + 1))
    for i = 1:k_r
        polynomial_r = derivative(polynomial_r)       #Differentiation up to k
    end

    polynomial_psi = Polynomial(ones(order + 1))
    for i = 1:k_psi
        polynomial_psi = derivative(polynomial_psi)   #Differentiation up to k
    end

    A = []
    for i = 1:m
        A_x = zeros(order + 1, order + 1)
        A_y = zeros(order + 1, order + 1)
        A_z = zeros(order + 1, order + 1)
        A_psi = zeros(order + 1, order + 1)
        for j = 1:order+1
            for k = j:order+1

                #Position
                if (j <= length(polynomial_r) && (k <= length(polynomial_r)))
                    order_t_r = ((order - k_r - j + 1) + (order - k_r - k + 1))
                    if (j == k)
                        A_x[j, k] =
                            polynomial_r[j-1]^2 / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_y[j, k] =
                            polynomial_r[j-1]^2 / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_z[j, k] =
                            polynomial_r[j-1]^2 / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                    else
                        A_x[j, k] =
                            2 * polynomial_r[j-1] * polynomial_r[k-1] / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_y[j, k] =
                            2 * polynomial_r[j-1] * polynomial_r[k-1] / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                        A_z[j, k] =
                            2 * polynomial_r[j-1] * polynomial_r[k-1] / (order_t_r + 1) *
                            (t[i+1]^(order_t_r + 1) - t[i]^(order_t_r + 1))
                    end
                end

                #Yaw
                if (j <= length(polynomial_psi) && (k <= length(polynomial_psi)))
                    order_t_psi = ((order - k_psi - j + 1) + (order - k_psi - k + 1))
                    if (j == k)
                        A_psi[j, k] =
                            polynomial_psi[j-1]^2 / (order_t_psi + 1) *
                            (t[i+1]^(order_t_psi + 1) - t[i]^(order_t_psi + 1))
                    else
                        A_psi[j, k] =
                            2 * polynomial_psi[j-1] * polynomial_psi[k-1] /
                            (order_t_psi + 1) *
                            (t[i+1]^(order_t_psi + 1) - t[i]^(order_t_psi + 1))
                    end
                end

            end
        end
        if i == 1
            blocks = [mu_r * A_x, mu_r * A_y, mu_r * A_z, mu_psi * A_psi]
        else
            blocks = [A, mu_r * A_x, mu_r * A_y, mu_r * A_z, mu_psi * A_psi]
        end
        A = ControlSystems.blockdiag(blocks...)
    end
    A = 0.5 * (A + A') #Make it symmetric
end


function computeConstraint(order, m, k_r, k_psi, t, keyframe, corridor_position, n_intermediate, corridor_width)

n = 4;                              #State number

#Waypoint constraints
C1 = zeros(2*m*n,n*(order+1)*m);
b1 = zeros(2*m*n);
computeMat = diagm(ones(order+1));          #Required for computation of polynomials
for i=1:m
    waypoint = keyframe[:,i];       #Waypoint at t(i)

    if(i==1)                        #Initial and Final Position
        #Initial
        values = zeros(1,order+1);
        for j=1:order+1
            poly = Polynomial(computeMat[j,:])
            values[j] = poly(t[i])
        end

        for k=1:n
            c = zeros(1,n*(order+1)*m);
            c[ ((i-1)*(order+1)*n+(k-1)*(order+1)+1) : ((i-1)*(order+1)*n+(k-1)*(order+1))+order+1 ] = values;
            C1[k,:] = c;
        end
        b1[1:n] = waypoint;

        #Final
        for j=1:order+1
            values(j) = polyval(computeMat(j,:),t(m+1));
        end

        for k=1:n
            c = zeros(1,n*(order+1)*m);
            c( ((m-1)*(order+1)*n+(k-1)*(order+1)+1) : ((m-1)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
            C1(k+n,:) = c;
        end
        b1(n+1:2*n) = waypoint;

    else
        #Elsewhere
        values = zeros(1,order+1);
        for j=1:order+1
            values(j) = polyval(computeMat(j,:),t(i));
        end

        for k=1:n
            c = zeros(1,n*(order+1)*m);
            c( ((i-2)*(order+1)*n+(k-1)*(order+1)+1) : ((i-2)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
            C1(k+2*n*(i-1),:) = c;
        end
        b1(2*n*(i-1)+1:2*n*(i-1)+n) = waypoint;

        for k=1:n
            c = zeros(1,n*(order+1)*m);
            c( ((i-1)*(order+1)*n+(k-1)*(order+1)+1) : ((i-1)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
            C1(k+2*n*(i-1)+n,:) = c;
        end
        b1(2*n*(i-1)+n+1:2*n*(i-1)+2*n) = waypoint;

    end

end


# Derivative constraints

# Position
C2 = zeros(2*m*(n-1)*k_r,n*(order+1)*m);                    #(n-1) : yaw excluded here
b2 = ones(2*m*(n-1)*k_r,1)*eps;
constraintData;
#constraintData_r = zeros(m,k_r,3);
for i=1:m
    for h=1:k_r
        if(i==1)
            #Initial
            values = zeros(1,order+1);
            for j=1:order+1
                tempCoeffs = computeMat(j,:);
                for k=1:h
                    tempCoeffs = polyder(tempCoeffs);
                end
                values(j) = polyval(tempCoeffs,t(i));
            end

            continuity = zeros(1,n-1);
            for k=1:n-1
                if(constraintData_r(i,h,k)==eps)
                    %Continuity
                    continuity(k) = true;
                end

                c = zeros(1,n*(order+1)*m);
                if(continuity(k))
                    c( ((i-1)*(order+1)*n+(k-1)*(order+1)+1) : ((i-1)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
                    c( ((m-1)*(order+1)*n+(k-1)*(order+1)+1) : ((m-1)*(order+1)*n+(k-1)*(order+1))+order+1) = -values;
                    C2(k + (h-1)*(n-1),:) = c;
                    b2(k + (h-1)*(n-1)) = 0;
                else
                    c( ((i-1)*(order+1)*n+(k-1)*(order+1)+1) : ((i-1)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
                    C2(k + (h-1)*(n-1),:) = c;
                    b2(k + (h-1)*(n-1)) = constraintData_r(i,h,k);
                end
            end

            #Final
            values = zeros(1,order+1);
            for j=1:order+1
                tempCoeffs = computeMat(j,:);
                for k=1:h
                    tempCoeffs = polyder(tempCoeffs);
                end
                values(j) = polyval(tempCoeffs,t(m+1));
            end

            for k=1:n-1
                if(constraintData_r(i,h,k)==eps)
                    #Continuity
                end
                c = zeros(1,n*(order+1)*m);
                if(~continuity(k))
                    c( ((m-1)*(order+1)*n+(k-1)*(order+1)+1) : ((m-1)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
                    C2(k + (h-1)*(n-1) + (n-1)*k_r,:) = c;
                    b2(k + (h-1)*(n-1) + (n-1)*k_r) = constraintData_r(i,h,k);
                end
            end

        else

            #Elsewhere
            values = zeros(1,order+1);
            for j=1:order+1
                tempCoeffs = computeMat(j,:);
                for k=1:h
                    tempCoeffs = polyder(tempCoeffs);
                end
                values(j) = polyval(tempCoeffs,t(i));
            end

            continuity = zeros(1,n-1);
            for k=1:n-1
                if(constraintData_r(i,h,k)==eps)
                    #Continuity
                    continuity(k) = true;
                end

                c = zeros(1,n*(order+1)*m);
                if(continuity(k))
                    c( ((i-2)*(order+1)*n+(k-1)*(order+1)+1) : ((i-2)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
                    c( ((i-1)*(order+1)*n+(k-1)*(order+1)+1) : ((i-1)*(order+1)*n+(k-1)*(order+1))+order+1) = -values;
                    C2(k + (h-1)*(n-1) + 2*(i-1)*(n-1)*k_r,:) = c;
                    b2(k + (h-1)*(n-1) + 2*(i-1)*(n-1)*k_r) = 0;
                else
                    c( ((i-2)*(order+1)*n+(k-1)*(order+1)+1) : ((i-2)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
                    C2(k + (h-1)*(n-1) + 2*(i-1)*(n-1)*k_r,:) = c;
                    b2(k + (h-1)*(n-1) + 2*(i-1)*(n-1)*k_r) = constraintData_r(i,h,k);
                end
            end

            continuity = zeros(1,n-1);
            for k=1:n-1
                if(constraintData_r(i,h,k)==eps)
                    #Continuity
                    continuity(k) = true;
                end
                c = zeros(1,n*(order+1)*m);

                if(~continuity(k))
                    c( ((i-1)*(order+1)*n+(k-1)*(order+1)+1) : ((i-1)*(order+1)*n+(k-1)*(order+1))+order+1) = values;
                    C2(k + (h-1)*(n-1) + 2*(i-1)*(n-1)*k_r + (n-1)*k_r,:) = c;
                    b2(k + (h-1)*(n-1) + 2*(i-1)*(n-1)*k_r + (n-1)*k_r) = constraintData_r(i,h,k);
                end

            end

        end
    end
end

#Corridor constraints

C3 = [];

b3 = [];
t_vector = (keyframe(1:3,corridor_position(2)) - keyframe(1:3,corridor_position(1)))...
/norm(keyframe(1:3,corridor_position(2)) - keyframe(1:3,corridor_position(1)));
#unit vector of direction of the corridor

t_intermediate = linspace(t(corridor_position(1)),t(corridor_position(2)),n_intermediate+2);
t_intermediate = t_intermediate(2:end-1);
#intermediate time stamps

computeMat = eye(order+1);          #Required for computation of polynomials
for i = 1:n_intermediate
    values = zeros(1,order+1);
    for j=1:order+1
        values(j) = polyval(computeMat(j,:),t_intermediate(i));
    end

    c = zeros(6, n*(order+1)*m);       #Absolute value constraint : two inequality constraints
    b = zeros(6, 1);

    rix = keyframe(1,corridor_position(1));
    riy = keyframe(2,corridor_position(1));
    riz = keyframe(3,corridor_position(1));
    #x
    c(1,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1))...
        = [values zeros(1,2*(order+1))]...
        - t_vector(1)*[t_vector(1)*values t_vector(2)*values t_vector(3)*values];
    b(1) = corridor_width +...
        rix+t_vector(1)*(-rix*t_vector(1) -riy*t_vector(2) -riz*t_vector(3));
    c(2,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1))...
        = -c(1,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1));
    b(2) = corridor_width +...
        -rix-t_vector(1)*(-rix*t_vector(1) -riy*t_vector(2) -riz*t_vector(3));
    #y
    c(3,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1))...
        = [zeros(1,order+1) values zeros(1,order+1)]...
        - t_vector(2)*[t_vector(1)*values t_vector(2)*values t_vector(3)*values];
    b(3) = corridor_width +...
        riy+t_vector(2)*(-rix*t_vector(1) -riy*t_vector(2) -riz*t_vector(3));
    c(4,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1))...
        = -c(3,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1));
    b(4) = corridor_width +...
        -riy-t_vector(2)*(-rix*t_vector(1) -riy*t_vector(2) -riz*t_vector(3));
    #z
    c(5,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1))...
        = [zeros(1,2*(order+1)) values]...
        - t_vector(3)*[t_vector(1)*values t_vector(2)*values t_vector(3)*values];
    b(5) = corridor_width +...
        riz+t_vector(3)*(-rix*t_vector(1) -riy*t_vector(2) -riz*t_vector(3));
    c(6,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1))...
        = -c(5,(corridor_position(1)-1)*n*(order+1)+0*(order+1)+1:(corridor_position(1)-1)*n*(order+1)+3*(order+1));
    b(6) = corridor_width +...
        -riz-t_vector(3)*(-rix*t_vector(1) -riy*t_vector(2) -riz*t_vector(3));

    C3 = [C3; c];
    b3 = [b3; b];
end

C = [C1; C2];
b = [b1; b2];
end
