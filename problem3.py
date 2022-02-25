import numpy as np
import matplotlib.pyplot as plt


def g(omega):
    gamma = 1
    return (gamma/np.pi)/(omega**2 + gamma**2)^(-1)


gamma = 1
K_c = 2*gamma
K = [K_c-1, K_c+0.01, K_c+1]
