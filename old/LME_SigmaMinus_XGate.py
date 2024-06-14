import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T = 4.551E-6
gamma = 500**2


def model_x(y,t):
    return -(gamma/2)*y

def model_y(y,t):
    return -(gamma/2)*y

def model_z(y,t):
    return (gamma/2) - (gamma*y)


y0 = 1
t = np.arange(0,T,T/100)

x1 = odeint(model_x,y0,t)
y1 = odeint(model_y,y0,t)
z1 = odeint(model_z,y0,t)

x2 = odeint(model_x,x1[-1],t[-1]+t)
y2 = odeint(model_y,y1[-1],t[-1]+t)
z2 = odeint(model_z,1-z1[-1],t[-1]+t)

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))
z = np.concatenate((z1,z2))

t = np.concatenate((t, t[-1]+t))

fig, ax = plt.subplots(2, 1)
fig.set_figheight(8)
fig.set_figwidth(8)
ax[0].plot(t,x,'.')
ax[0].set_title('X/Y')
ax[0].legend([r'$\sigma_{-}$'])
ax[1].plot(t,z,'.')
ax[1].set_title('Z')
ax[1].legend([r'$\sigma_{-}$'])
plt.savefig('LME_SIGMA-_ODE_Solver_XGate.pdf')
plt.show()
