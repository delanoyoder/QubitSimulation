import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T = 1E-1
gamma = 100


def model_x(y,t):
    return -(gamma/2)*y

def model_y(y,t):
    return -(gamma/2)*y

def model_z(y,t):
    return (gamma/2) - (gamma*y)


y0 = 1
t = np.arange(0,T,T/100)

x = odeint(model_x,y0,t)
y = odeint(model_y,y0,t)
z = odeint(model_z,y0,t)

M_x = np.exp(-(gamma/2)*t)
M_y = np.exp(-(gamma/2)*t)
M_z = np.exp(-gamma*t) + (1-np.exp(-gamma*t))/2

fig, ax = plt.subplots(2, 1)
fig.set_figheight(8)
fig.set_figwidth(8)
ax[0].plot(t,x,'.')
ax[0].plot(t,M_x,'--')
ax[0].set_title('X/Y')
ax[0].legend([r'$\sigma_{-}$', 'ODE Solver'])
ax[1].plot(t,z,'.')
ax[1].plot(t,M_z,'--')
ax[1].set_title('Z')
ax[1].legend([r'$\sigma_{-}$', 'ODE Solver'])
plt.savefig('ODE_Solver_Test.pdf')
plt.show()
