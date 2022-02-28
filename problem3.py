import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


gamma = 0.1
K_c = 2*gamma
Ks = [0.01, (K_c+0.01), (K_c + 2)]
N = 100
theta_0 = np.random.uniform(-np.pi/2, np.pi/2, size=(3, N))
theta = theta_0.copy()
omega = stats.cauchy.rvs(loc=0, scale=gamma, size=(3, N))
T = 100
dt = 0.1
r = np.zeros((3, int(T/dt)))
t_vec = np.linspace(0, T, int(T/dt))

for k, a in enumerate(Ks):
    for t in range(int(T/dt)):
        for i in range(N):
            theta[k, i] += dt*(omega[k, i] + (a * np.sum(np.sin(theta[k, :] - theta[k, i]))/N))

        r[k, t] = np.abs(np.sum(np.exp(1j*theta[k, :]))/N)


plt.subplot(331)
plt.title(f'Order parameter, K={Ks[0]}, N={N}')
plt.plot(t_vec, r[0, :])
plt.xlabel('t')
plt.ylabel('r')
plt.ylim([0, 1])
plt.subplot(332)
plt.title(f'Oscillators at time={T}, dt={dt}')
plt.plot(np.cos(theta[0, :]), np.sin(theta[0, :]), 'o', markersize=2)
plt.xlabel(r'$ cos(\Theta) $')
plt.ylabel(r'$ sin(\Theta) $')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.subplot(333)
plt.title('Oscillators at time=0')
plt.plot(np.cos(theta_0[0, :]), np.sin(theta_0[0, :]), 'o', markersize=2)
plt.xlabel(r'$ cos(\Theta) $')
plt.ylabel(r'$ sin(\Theta) $')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.subplot(334)
plt.title(f'Order parameter, K={0.21}, N={N}')
plt.plot(t_vec, r[1, :])
plt.xlabel('t')
plt.ylabel('r')
plt.ylim([0, 1])
plt.subplot(335)
plt.title(f'Oscillators at time={T}, dt={dt}')
plt.plot(np.cos(theta[1, :]), np.sin(theta[1, :]), 'o', markersize=2)
plt.xlabel(r'$ cos(\Theta) $')
plt.ylabel(r'$ sin(\Theta) $')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.subplot(336)
plt.title('Oscillators at time=0')
plt.plot(np.cos(theta_0[1, :]), np.sin(theta_0[1, :]), 'o', markersize=2)
plt.xlabel(r'$ cos(\Theta) $')
plt.ylabel(r'$ sin(\Theta) $')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.subplot(337)
plt.title(f'Order parameter, K={Ks[2]}, N={N}')
plt.plot(t_vec, r[2, :])
plt.xlabel('t')
plt.ylabel('r')
plt.ylim([0, 1])
plt.subplot(338)
plt.title(f'Oscillators at time={T}, dt={dt}')
plt.plot(np.cos(theta[2, :]), np.sin(theta[2, :]), 'o', markersize=2)
plt.xlabel(r'$ cos(\Theta) $')
plt.ylabel(r'$ sin(\Theta) $')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.subplot(339)
plt.title('Oscillators at time=0')
plt.plot(np.cos(theta_0[2, :]), np.sin(theta_0[2, :]), 'o', markersize=2)
plt.xlabel(r'$ cos(\Theta) $')
plt.ylabel(r'$ sin(\Theta) $')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.tight_layout(pad=0, w_pad=-0.8, h_pad=-0.5)
plt.show()
