import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T = 4.551E-6
gamma = 500**2


def model_x(y,t):
    return -(2*gamma)*y

def model_y(y,t):
    return -(2*gamma)*y

def model_z(y,t):
    return y*0


y0 = 1
t = np.arange(0,T,T/100)

x1 = odeint(model_x,y0,t)
y1 = odeint(model_y,y0,t)
z1 = odeint(model_z,y0,t)

x2 = odeint(model_x,x1[-1],t[-1]+t)
y2 = odeint(model_y,-y1[-1],t[-1]+t)
z2 = odeint(model_z,-z1[-1],t[-1]+t)

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))
z = np.concatenate((z1,z2))

t = np.concatenate((t, t[-1]+t))

fig, ax = plt.subplots(3, 1)
fig.set_figheight(8)
fig.set_figwidth(8)
ax[0].plot(t,x,'.')
ax[0].set_title('X')
ax[0].legend([r'$\sigma_{z}$'])
ax[1].plot(t,y,'.')
ax[1].set_title('Y')
ax[1].legend([r'$\sigma_{z}$'])
ax[2].plot(t,z,'.')
ax[2].set_title('Z')
ax[2].legend([r'$\sigma_{z}$'])
plt.savefig('LME_SIGMAZ_ODE_Solver_XGate.pdf')
plt.show()
