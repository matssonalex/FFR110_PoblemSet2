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


def func(u, rho, q_val, i, tau):
    if i == 0 or i == 99:
        return rho * u[i, tau] * (1 - (u[i, tau] / q_val)) - (u[i, tau] / (1 + u[i, tau]))
    else:
        return rho * u[i, tau] * (1 - (u[i, tau] / q_val)) - (u[i, tau] / (1 + u[i, tau]))+ u[i + 1, tau] + u[i - 1, tau] - 2*u[i, tau]


def eom(u, dt, rho, q_val, i, tau):
    return u[i, tau] + (dt * func(u, rho, q_val, i, tau))


def estimate_velocity(u, dt, t, alt):
    u_val = max(u[:, t]) / 2
    
    if alt == 0:
        ind1 = np.where(u[:, t] < u_val)[0][0]
        ind2 = t + np.where(u[ind1+1, t:] > u_val)[0][0]
    elif alt == 1: 
        ind1 = np.where(u[:, t] < u_val)[0][0]
        ind2 = np.where(u[ind1-1, :] < u_val)[0][0]
    elif alt == 2:
        ind1 = np.where(u[:, t] < u_val)[0][0]
        ind2 = t + np.where(u[ind1+1, t:] > u_val)[0][0]

    v = 1 / (dt * (ind2-t))
    return v


def run_dynamics(xi0, u0, rho, q_val, dt, t_max, L, init_func):
    u = np.zeros((L, int(t_max/dt)))
    
    if init_func == "ramp":
        init_ramp_func(xi0, u0, u)
    elif init_func == "peak":
        init_peak_func(xi0, u0, u)

    for tau in range(int((t_max/dt)) - 1):
        for i_xi in range(L):
            u[i_xi, tau + 1] =  eom(u, dt, rho, q_val, i_xi, tau)
    
    return u


def b_plot(u, xi, derivative_estimate, T, dt):
    # t_vec = np.linspace(0, np.shape(u)[1]-1, 20)
    # for t in t_vec:
    #     plt.plot(xi, u[:, int(t)])
    #     plt.xlabel(r'$\xi$')
    #     plt.ylabel('u')
  
    plt.subplot(121)
    plt.plot(xi, u[:, T])
    plt.xlabel(r'$\xi$')
    plt.ylabel('u')

    plt.subplot(122)
    plt.plot(derivative_estimate, u[:-1, T], 'b.')
    plt.xlabel(r'$\frac{du}{d\xi}$')
    plt.ylabel("u")

    plt.suptitle(f"Travelling wave at t = {T*dt}")



def c_plot(u1, u2, xi):
    t_vec = np.linspace(0, np.shape(u1)[1]-1, 10)
    for t in t_vec:
        ax1 = plt.subplot(121)
        ax1.plot(xi, u1[:, int(t)])
        ax2 = plt.subplot(122, sharey=ax1)
        ax2.plot(xi, u2[:, int(t)])
    
    ax1.set_ylabel('u')
    ax1.set_xlabel(r'$\xi$')
    ax1.set_title(r"$u_0 = u_1^*$")
    ax2.set_xlabel(r'$\xi$')
    ax2.set_title(r"$u_0 = 3 \cdot u_1^*$")
    plt.suptitle(r"Dynamics with initial peak at $\xi_0$=50")


def main(alt, dyn):
    L = 100
    t_max = 500
    dt = 0.01
    rho = 0.5
    q_val = 8
    xi = np.arange(1, L + 1)
    
    u_star_val = find_steady_states(q_val, rho) #u1 = u_star_val[2], u2 = u_star_val[1]
    print(u_star_val)
    
    if dyn == 'b':
        T = [1000, 1000, 5000]
        xi0 = [20, 50, 50]
        u0 = [u_star_val[2], u_star_val[1], 1.1*u_star_val[1]]
        u_b = run_dynamics(xi0[alt], u0[alt], rho, q_val, dt, t_max, L, "ramp")  #t=1000, xi0= 20, 50, 50 u0 = u1, u2, u2*1.1

        v_estimate = estimate_velocity(u_b, dt, T[alt], alt) # only works if u is a travelling wave
        print(f"Wave velocity = {v_estimate}")

        derivative_estimate = np.diff(u_b[:, T[alt]])
        b_plot(u_b, xi, derivative_estimate, T[alt], dt)
    
    elif dyn == 'c':
        u_c1 = run_dynamics(50, u_star_val[2], rho, q_val, dt, t_max, L, "peak")
        u_c2 = run_dynamics(50, u_star_val[2]*3, rho, q_val, dt, t_max, L, "peak")
        c_plot(u_c1, u_c2, xi)


    plt.show()

if __name__ == "__main__":
    alt = 0 # 0, 1 or 2 for the 3 cases, b or c for different dynamics.
    main(alt, 'c')