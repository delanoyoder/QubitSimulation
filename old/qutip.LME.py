import numpy as np
from qutip import *
import matplotlib.pyplot as plt

w = 5E9 # 5 Ghz
h = 1.05E-34 # 1.0545718×10^(-34) m^2 kg / s
b = w*h/2
gamma = 100**2

H = -b * sigmaz()

psi0 = basis(2, 0)

times = np.arange(0,1E-3,1E-6)

result = mesolve(H, psi0, times, [np.sqrt(gamma) * sigmap()], [sigmaz()])

fig, ax = plt.subplots()

ax.plot(result.times, result.expect[0]);

ax.set_xlabel('Time');

ax.set_ylabel('Expectation values');

ax.legend(("Sigma-Z"));

plt.show()