import numpy as np
import matplotlib.pyplot as plt

i = complex(0,1)
w = 5E9 # 5 Ghz
h = 1.05E-34 # 1.0545718×10^(-34) m^2 kg / s
b = w*h/2
gamma = 100**2

I = np.array([[1,0],
              [0,1]])
X = np.array([[0,1],
              [1,0]])
Y = np.array([[0,-i],
              [i,0]])
Z = np.array([[1,0],
              [0,-1]])

P =[0.5*I, X, Y, Z]

H = -b*Z

def LME(T,L):
    
    dp = []
    
    for t in T:
        
        dpdt = np.array([[0,0],
                         [0,0]])
        
        for p in P:

            A = np.dot(np.dot(np.exp(i*H*t),L),np.exp(-i*H*t))
            B = np.dot(np.dot(np.exp(i*H*t),p),np.exp(-i*H*t))
            C = np.dot(np.dot(np.exp(i*H*t),np.conj(np.transpose(L))),np.exp(-i*H*t))
            
            dpdt = dpdt + np.dot(np.dot(A,B),C)
            dpdt = dpdt - 0.5*np.dot(np.dot(C,A),B)
            dpdt = dpdt - 0.5*np.dot(np.dot(B,C),A)

        dpdt = dpdt - 0.5*I
        dp.append(dpdt)
        
        print(dpdt[0][0] == -dpdt[1][1])
            
    return dp

T = np.arange(0,10,1)
L = np.sqrt(gamma)*np.array([[0,1],[0,0]])

LME = LME(T,L)

#plt.plot(T,LME)
#plt.show()