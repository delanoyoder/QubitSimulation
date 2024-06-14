function varargout=LME(t,rho0,L)
%% Solve the Lindblad Master Equation
%  ...=LME(t,p,H,L) solves the evolution of a quantum system with a
%  constant Lindbladian superoperator L, given the initial density matrix rho0.
%%
I = [1 0;0 1]
X = [0 1;1 0]
Y = [0 -i;i 0]
Z = [1 0;0 -1]

H = -Z

p = {0.5*I, X, Y, Z}

dpdt = [0 0;0 0]

sz = size(t)

for j=sz(1):sz(2)
    for k=1:4
        A = exp(i*H*t(j))*L*exp(-i*H*t(j))
        B = exp(i*H*t(j))*p{k}*exp(-i*H*t(j))
        C = exp(i*H*t(j))*L'*exp(-i*H*t(j))
        dpdt = dpdt + A*B*C - 0.5*C*B*A - 0.5*A*C*B
    end
end




