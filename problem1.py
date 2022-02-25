from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import derivative
import sympy as sym
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})


def find_steady_states(q_val, rho):
    u, q, p = sym.symbols('u q p')
    fp = sym.solve(p*u*(1 - (u/q)) - (u/(1 + u)), u)
    print(fp)

    u_star1_val = fp[0].subs([(q, q_val), (p, rho)])
    u_star2_val = fp[1].subs([(q, q_val), (p, rho)])
    u_star3_val = fp[2].subs([(q, q_val), (p, rho)])
    
    return [u_star1_val, u_star2_val, u_star3_val]


def init_ramp_func(xi0, u0, u):
    for i in range(np.shape(u)[0]):
        u[i, 0] = u0 / (1 + np.exp((i+1) - xi0))


def init_peak_func(xi0, u0, u):
    for i in range(np.shape(u)[0]):
        u[i, 0] = u0 * np.exp(-(((i+1) - xi0)**2))


# def disc_lap(u, i, tau):
#     return u[i + 1, tau] + u[i - 1, tau] - 2*u[i, tau]


def func(u, i, tau):
    if i == 0 or i == 99:
        return rho * u[i, tau] * (1 - (u[i, tau] / q_val)) - (u[i, tau] / (1 + u[i, tau]))
    else:
        return rho * u[i, tau] * (1 - (u[i, tau] / q_val)) - (u[i, tau] / (1 + u[i, tau])) + u[i + 1, tau] + u[i - 1, tau] - 2*u[i, tau]


def eom(u, i, tau):
    return u[i, tau] + (dt * func(u, i, tau))


def estimate_velocity(u, t):
    ind = np.where(np.abs(u[24, t] - u[34,:]) < 1e-3)[0][0]
    v = 1 / (dt * (ind-t))
    return v


def run_dynamics(xi0, u0, init_func):
    u = np.zeros((L, int(t_max/dt)))
    
    if init_func == 1:
        init_ramp_func(xi0, u0, u)
    elif init_func == 2:
        init_peak_func(xi0, u0, u)

    # u[0, :] = u[0, 0]
    # u[-1, :] = u[-1, 0] 

    for tau in range(int((t_max/dt)) - 1):
        for i_xi in range(L):
            u[i_xi, tau + 1] =  eom(u, i_xi, tau)
    
    # take care of boundaries, always same
    return u


def b_plot(u, derivative_estimate, T):
    t_vec = np.linspace(0, np.shape(u)[1]-1, 20)
    # # # t_vec = [1000, 2000]
    for t in t_vec:
        plt.plot(xi, u[:, int(t)])
        plt.xlabel(r'$\xi$')
        plt.ylabel('u')
    
    plt.title(f'Case iii')
    # plt.subplot(121)
    # plt.plot(xi, u[:, T])
    # plt.xlabel(r'$\xi$')
    # plt.ylabel('u')

    # plt.subplot(122)
    # plt.plot(derivative_estimate, u[:-1, T], 'b.')
    # plt.xlabel(r'$\frac{du}{d\xi}$')
    # plt.ylabel("u")

    # plt.suptitle(f"Travelling wave at t = {T*dt}")



def c_plot(u1, u2):
    # t_vec = np.linspace(1, int(t_max/dt)-1, 10)
    t_vec = np.linspace(1000, 10000, 10)

    # t_vec = np.linspace(0,, 10)
    for t in t_vec:
        plt.subplot(121)
        plt.plot(xi, u1[:, int(t)])
        plt.subplot(122)
        plt.plot(xi, u2[:, int(t)])



L = 100
t_max = 500
dt = 0.01
rho = 0.5
q_val = 8
xi = np.arange(1, L + 1)
u_star_val = find_steady_states(q_val, rho) #u1 = u_star_val[2], u2 = u_star_val[1]
print(u_star_val)

T = 1000
xi0 = 50
u0 = u_star_val[1]*1.1
u_b = run_dynamics(xi0, u0, 1)    #t=1000, xi0= 20, 50, 50 u0 = u1, u2, u2*1.1

# v_estimate = estimate_velocity(u_b, T) # only works if u is a travelling wave
# print(v_estimate)

derivative_estimate = np.diff(u_b[:, T])
b_plot(u_b, derivative_estimate, T)

# u_c1 = run_dynamics(50, u_star_val[2], 2)
# u_c2 = run_dynamics(50, u_star_val[2]*3, 2)
# c_plot(u_c1, u_c2)


plt.show()
