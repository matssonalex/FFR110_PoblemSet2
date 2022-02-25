import numpy as np
import sympy as sym
from matplotlib import pyplot as plt


# a, b, u, v = sym.symbols("a b u v")
# sol = sym.solve([a - (b+1)*u + (u**2)*v, b*u - (u**2)*v], (u, v))
# print(sol)
def disc_lap(f, x, y):
    L = len(f[:, 0])
    return f[(x + 1) % L, y] + f[(x - 1) % L, y] + f[x, (y + 1) % L] + f[x, (y - 1) % L] - 4*f[x, y]
    

def fu(u_grid, v_grid, x, y, a, b, Du):
    return a - (b+1)*u_grid[x,y] + (u_grid[x,y]**2)*v_grid[x,y] + Du*(disc_lap(u_grid,x,y))


def fv(u_grid, v_grid, x, y, b, Dv):
    return b*u_grid[x,y] - (u_grid[x,y]**2)*v_grid[x,y] + Dv*(disc_lap(v_grid,x,y))   


def eom(u_grid, v_grid, x, y, dt, a, b, Du, Dv):
    new_x = u_grid[x,y] + dt * fu(u_grid, v_grid, x, y, a, b, Du)
    new_y = v_grid[x,y] + dt * fv(u_grid, v_grid, x, y, b, Dv)
    return new_x, new_y
    

def main():
    a = 3
    b = 8
    Du = 1
    Dv_vec = [2.3, 3, 5, 9]
    Dv = 2.3
    L = 128
    t_max = 100
    dt = 0.01


    # plt.figure()

    u_grid = (1.1*a - 0.9*a) * np.random.rand(L, L) + 0.9*a
    v_grid = (1.1*b/a - 0.9*b/a) * np.random.rand(L, L) + 0.9*b/a
    
    for iter in range(int(t_max/dt)):
        # if iter % 1000 == 0:
        #     plt.imshow(u_grid)
        #     plt.title(f"{Dv=} at iteration: {iter}, t = {iter*dt}")
        #     plt.show()
        
        for x in range(L):
            for y in range(L):
                updated_u, updated_v = eom(u_grid, v_grid, x, y, dt, a, b, Du, Dv)
                u_grid[x,y] = updated_u
                v_grid[x,y] = updated_v

    plt.imshow(u_grid)
    plt.title(f"{Dv=} at iteration: {iter}, t = {iter*dt}")
    plt.show()            


# JUST FOR TEST
if __name__ == "__main__":
    main()
    