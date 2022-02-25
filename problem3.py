import numpy as np
import matplotlib.pyplot as plt


def circle(vector):
    x = []
    y = []
    for angle in vector:
        x.append(np.cos(angle))
        y.append(np.sin(angle))
    plt.scatter(x, y)
    plt.axes().set_aspect( 1 )
    plt.show()





# def g(omega):
#     gamma = 1
#     return (gamma/np.pi)/(omega**2 + gamma**2)


gamma = 1
K_c = 2*gamma
K = [K_c-1, K_c+0.01, K_c+1]
K = K[0]
N = 30
theta = np.random.uniform(np.pi/2, -np.pi/2, size=N)
omega = np.random.standard_cauchy(size=N)
T = 100
dt = 0.01
r = np.zeros(int(T/dt))
t = np.linspace(0, T, int(T/dt))

for j in range(int(T/dt)):
    for i in range(N):
        r[j] = np.sqrt(1 + np.sum(np.cos(theta - theta[i])))
        #r[j] = np.abs(np.sum(np.exp(theta)))
        theta[i] += dt*(omega[i] + K/N*np.sum(np.sin(theta - theta[i])))

#circle(theta)
plt.plot(t, r)
plt.show()
