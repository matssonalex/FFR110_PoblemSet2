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
    new_u = u_grid[x,y] + dt * fu(u_grid, v_grid, x, y, a, b, Du)
    new_v = v_grid[x,y] + dt * fv(u_grid, v_grid, x, y, b, Dv)
    return new_u, new_v
    

def main():
    a = 3
    b = 8
    Du = 1
    Dv_vec = [2.3, 3, 5, 9]
    L = 128
    t_max = 1.5
    dt = 0.01

    max_u = 0
    min_u = 10

    # fig1 = plt.figure(figsize=(14,8))
    # fig2 = plt.figure(figsize=(14,8))
    
    
    # for ind, Dv in enumerate(Dv_vec):

    u_grid = (1.1*a - 0.9*a) * np.random.rand(L, L) + 0.9*a
    v_grid = (1.1*b/a - 0.9*b/a) * np.random.rand(L, L) + 0.9*b/a
    Dv = 5
    # ax1 = fig1.add_subplot(2, 2, ind + 1)
    # im1 = ax1.imshow(u_grid, cmap="viridis", vmin=0, vmax=12)
    # fig1.colorbar(im1)

    for iter in range(int(t_max/dt)+1):
        
        if iter % 2500 == 0:
            print(f"{Dv=}, {iter=} of {int(t_max/dt)}")
        
        for x in range(L):
            for y in range(L):
                updated_u, updated_v = eom(u_grid, v_grid, x, y, dt, a, b, Du, Dv)
                u_grid[x,y] = updated_u
                v_grid[x,y] = updated_v

    
        # if iter == 10:
        #     ax1.imshow(u_grid, cmap="viridis", vmin=0, vmax=12, interpolation="nearest")
        #     ax1.set_title(f"{Dv=} at {iter=}")
        #     ax1.set_axis_off()
        #     fig1.tight_layout()


    
    # ax2 = fig2.add_subplot(2, 2, ind + 1)
    # im2 = ax2.imshow(u_grid, cmap="viridis", vmin=0, vmax=12, interpolation="nearest")
    # fig2.colorbar(im2)
    # ax2.set_title(f"{Dv=} at {iter=}")
    # ax2.set_axis_off()
    # fig2.tight_layout()
    plt.imshow(u_grid)
    plt.title(f"{Dv=} at iteration: {iter}, t = {iter*dt}")
    # fileName = f"Dv{ind}_long"
    # plt.savefig(fileName)

# fig1.savefig("short_run")
# fig2.savefig("long_run") 

    plt.show()


# JUST FOR TESTING 
if __name__ == "__main__":
    main()
  