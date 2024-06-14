import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T = 18.204E-6
g = 100**2
l = 100**2


def model_x(y,t):
    return -(g+4*l)*y

def model_y(y,t):
    return -(g+4*l)*y

def model_z(y,t):
    return -2*g*(y-1/2)

t = np.arange(0,T,T/100)

x1 = odeint(model_x, 0, t)
y1 = odeint(model_y, 0, t)
z1 = odeint(model_z, -1/2, t)

x2 = odeint(model_x, x1[-1], t[-1]+t)
y2 = odeint(model_y, y1[-1], t[-1]+t)
z2 = odeint(model_z, z1[-1], t[-1]+t)

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))
z = np.concatenate((z1,z2))

t = np.concatenate((t, t[-1]+t))

I = np.array([[1, 0],
              [0, 1]])

X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -complex(0, 1)],
              [complex(0, 1), 0]])

Z = np.array([[1, 0],
              [0, -1]])

x_est = []
y_est = []
z_est = []

for n in range(len(x)):
    
    P = I/2 + x[n]*X + y[n]*Y + z[n]*Z
    
    x_est.append((np.trace(np.dot(X, P))+1)/2)
    y_est.append((np.trace(np.dot(Y, P))+1)/2)
    z_est.append((np.trace(np.dot(Z, P))+1)/2)

fig, ax = plt.subplots(3, 1)
fig.set_figheight(11)
fig.set_figwidth(8.5)
ax[0].plot(t,x_est,'b')
ax[0].set_ylim(-0.05,1.05)
ax[0].legend(['X-axis'])
ax[1].plot(t,y_est,'r')
ax[1].set_ylim(-0.05,1.05)
ax[1].legend(['Y-axis'])
ax[2].plot(t,z_est,'g')
ax[2].set_ylim(-0.05,1.05)
ax[2].legend(['Z-axis'])
plt.savefig('LME_AVG_T1_T2_XGate.pdf')
plt.show()
