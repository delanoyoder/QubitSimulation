import numpy as np
import matplotlib.pyplot as plt

gamma = 100

T = 1E-1

t = np.arange(0,T,T/100)

M_x = np.exp(-0.5*gamma*t)
M_y = np.exp(-0.5*gamma*t)
M_z = np.exp(-gamma*t)+0.5*(1-np.exp(-gamma*t))

Z_x = np.exp(-2*gamma*t)
Z_y = np.exp(-2*gamma*t)
Z_z = np.exp(0*t)

fig, ax = plt.subplots(3, 1)
fig.set_figheight(8)
fig.set_figwidth(8)
ax[0].plot(t,M_x)
ax[0].plot(t,Z_x)
ax[0].set_title('X')
ax[0].legend([r'$\sigma_{-}$', r'$\sigma_{z}$'])
ax[1].plot(t,M_y)
ax[1].plot(t,Z_y)
ax[1].set_title('Y')
ax[1].legend([r'$\sigma_{-}$', r'$\sigma_{z}$'])
ax[2].plot(t,M_z)
ax[2].plot(t,Z_z)
ax[2].set_title('Z')
ax[2].legend([r'$\sigma_{-}$', r'$\sigma_{z}$'])
plt.show()