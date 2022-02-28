import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def circle(vector):
    x = []
    y = []
    for angle in vector:
        x.append(np.cos(angle))
        y.append(np.sin(angle))
    plt.scatter(x, y)
    # plt.axes().set_plt.aspect( 1 )
    #plt.show()


gamma = 0.5
K_c = 2*gamma
K = [K_c-0.1, K_c+0.01, K_c+0.1]
K = K[2]
N = 300
theta = np.random.uniform(-np.pi/2, np.pi/2, size=N)
# omega = np.random.standard_cauchy(size=N)
omega = stats.cauchy.rvs(loc=0, scale=gamma, size=N)
T = 100
dt = 0.1
r = np.zeros(int(T/dt))
t = np.linspace(0, T, int(T/dt))


for j in range(int(T/dt)):
    r[j] = abs(np.sum(np.exp(theta * 1j)) / N)
    for i in range(N):
        #r[j] = np.abs(np.sum(np.exp(theta)))

        theta[i] += dt*(omega[i] + ((K/N) * np.sum(np.sin(theta - theta[i])))) 
        # theta[i] += dt * (omega[i] + (K*r[j]*(np.mean(np.sin(theta - theta[i])))))
        if theta[i] < 0:
            theta[i] = theta[i] % (-2*np.pi)
        else:
            theta[i] = theta[i] % (2*np.pi)

    # temp = np.sum(theta[0] - theta[1:])
    # r[j] = sum([(np.exp(1j * i)) for i in theta])
    # r[j] = abs(r[j] / N)
    
    r1 = abs(np.mean(np.exp(theta * 1j)))
    # r[j] = np.sqrt(1 + np.cos(np.sum(theta[0] - theta)))
    # r[j] = np.linalg.norm([(np.sum(np.cos(theta)) / N), (np.sum(np.sin(theta)) / N)])
#circle(theta)
# temp = np.sqrt((K - K_c) / K_c)
# print(temp)
plt.plot(t, r)
plt.show()
