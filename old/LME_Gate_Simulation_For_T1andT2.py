import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

dt = 3.555556E-8
gate_t = np.arange(0, dt+1E-10, dt)
wait_t = np.arange(0, dt*2**7+1E-10, dt)
g = 100**2
l = 100**2


def model_x(y,t):
    return -(g+4*l)*y

def model_y(y,t):
    return -(g+4*l)*y

def model_z(y,t):
    return -2*g*(y-1/2)

def Initialize(x_0,y_0,z_0,t):
    
    x = odeint(model_x, x_0, t)
    y = odeint(model_y, y_0, t)
    z = odeint(model_z, z_0, t)
    
    T = t
    
    return T, x, y, z

def IGate(T,x,y,z,t):
    
    x_t = odeint(model_x, x[-1], t)
    y_t = odeint(model_y, y[-1], t)
    z_t = odeint(model_z, z[-1], t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def XGate(T,x,y,z,t):
    
    x_t = odeint(model_x, x[-1], t)
    y_t = odeint(model_y, -y[-1], t)
    z_t = odeint(model_z, -z[-1], t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def YGate(T,x,y,z,t):
    
    x_t = odeint(model_x, -x[-1], t)
    y_t = odeint(model_y, y[-1], t)
    z_t = odeint(model_z, -z[-1], t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def ZGate(T,x,y,z,t):
    
    x_t = odeint(model_x, -x[-1], t)
    y_t = odeint(model_y, -y[-1], t)
    z_t = odeint(model_z, z[-1], t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def HGate(T,x,y,z,t):
    
    x_t = odeint(model_x, z[-1]/np.sqrt(2), t)
    y_t = odeint(model_y, -y[-1]/np.sqrt(2), t)
    z_t = odeint(model_z, x[-1]/np.sqrt(2), t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def TGate(T,x,y,z,t):
    
    x_t = odeint(model_x, x[-1]/np.sqrt(2) - y[-1]/np.sqrt(2), t)
    y_t = odeint(model_y, x[-1]/np.sqrt(2) + y[-1]/np.sqrt(2), t)
    z_t = odeint(model_z, z[-1]/np.sqrt(2), t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def TdgGate(T,x,y,z,t):
    
    x_t = odeint(model_x, x[-1]/np.sqrt(2) + y[-1]/np.sqrt(2), t)
    y_t = odeint(model_y, -x[-1]/np.sqrt(2) + y[-1]/np.sqrt(2), t)
    z_t = odeint(model_z, z[-1]/np.sqrt(2), t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def SGate(T,x,y,z,t):
    
    x_t = odeint(model_x, -y[-1], t)
    y_t = odeint(model_y, x[-1], t)
    z_t = odeint(model_z, z[-1], t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z

def SdgGate(T,x,y,z,t):
    
    x_t = odeint(model_x, y[-1], t)
    y_t = odeint(model_y, -x[-1], t)
    z_t = odeint(model_z, z[-1], t)
    
    x = np.concatenate((x, x_t))
    y = np.concatenate((y, y_t))
    z = np.concatenate((z, z_t))
    
    T = np.concatenate((T, T[-1]+t))
    
    return T, x, y, z


def State_Estimation(x,y,z):
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
        
    return x_est, y_est, z_est

T, x, y, z = Initialize(0, 0, -1/2, gate_t)
T, x, y, z = XGate(T, x, y, z, gate_t)
T, x, y, z = IGate(T, x, y, z, wait_t)
T, x, y, z = YGate(T, x, y, z, gate_t)
T, x, y, z = IGate(T, x, y, z, wait_t)
T, x, y, z = ZGate(T, x, y, z, gate_t)
T, x, y, z = IGate(T, x, y, z, wait_t)
T, x, y, z = HGate(T, x, y, z, gate_t)
T, x, y, z = IGate(T, x, y, z, wait_t)
T, x, y, z = TGate(T, x, y, z, gate_t)
T, x, y, z = IGate(T, x, y, z, wait_t)
T, x, y, z = SGate(T, x, y, z, gate_t)
T, x, y, z = IGate(T, x, y, z, wait_t)
x, y, z = State_Estimation(x,y,z)

fig, ax = plt.subplots(3, 1)
fig.set_figheight(11)
fig.set_figwidth(8.5)
ax[0].plot(T,x,'b')
ax[0].set_ylim(-0.05,1.05)
ax[0].legend(['X-axis'])
ax[1].plot(T,y,'r')
ax[1].set_ylim(-0.05,1.05)
ax[1].legend(['Y-axis'])
ax[2].plot(T,z,'g')
ax[2].set_ylim(-0.05,1.05)
ax[2].legend(['Z-axis'])
plt.savefig('LME_AVG_T1_T2_XGate.pdf')
plt.show()
