import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.integrate import solve_ivp

# model parameters
du = 0.01
dv = 0.03
L = 1.0
Nx = 50 # number of patches
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]
K1=500
K2=500
# "same patch", "separate patch A", "separate patch B"
initial_condition="same patch"

t_span = (0, 150)
t_eval = np.linspace(t_span[0], t_span[1], 300)

K = np.zeros(Nx) 
K[:Nx//2] = K1
K[Nx//2:] = K2

# Initial conditions
u0 = np.zeros(Nx)
v0 = np.zeros(Nx)

if (initial_condition=="same patch"):    
    u0[:Nx//2] = 3
    v0[:Nx//2] = 3

if (initial_condition=="separate patch A"):    
    u0[:Nx//2] = 3
    v0[Nx//2:] = 3

if (initial_condition=="separate patch B"):
    u0[Nx//2:] = 3
    v0[:Nx//2] = 3

# PDE system
def pde_system(t, y):
    u = y[:Nx]
    v = y[Nx:]

    # diffusion terms
    u_xx = np.zeros_like(u)
    v_xx = np.zeros_like(v)

    # inner nodes
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    v_xx[1:-1] = (v[2:] - 2*v[1:-1] + v[:-2]) / dx**2

    # boundary nodes (zero-flux)
    u_xx[0] = (u[1] - u[0]) / dx**2
    u_xx[-1] = (u[-2] - u[-1]) / dx**2
    v_xx[0] = (v[1] - v[0]) / dx**2
    v_xx[-1] = (v[-2] - v[-1]) / dx**2

    # growth terms
    growth_u = u * (1 - (u + v) / K)
    growth_v = v * (1 - (u + v) / K)

    du_dt = du * u_xx + growth_u
    dv_dt = dv * v_xx + growth_v

    return np.concatenate([du_dt, dv_dt])

# initial population
y0 = np.concatenate([u0, v0])

# solve
sol = solve_ivp(pde_system, t_span, y0, t_eval=t_eval)

# show animation
U = sol.y[:Nx, :]
V = sol.y[Nx:, :]

def create_animation():
    fig, ax = plt.subplots(figsize=(12,6))
    line_u, = ax.plot([], [], label='u(x,t)', linewidth=2.5)
    line_v, = ax.plot([], [], label='v(x,t)', linewidth=1.5)
    ax.set_xlim(0, L)
    ax.set_ylim(0, np.max([U.max(), V.max()]) * 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('Population')
    ax.set_title('Population over time')
    ax.grid()
    ax.legend()

    def init():
        line_u.set_data([], [])
        line_v.set_data([], [])
        return line_u, line_v

    def animate(i):
        line_u.set_data(x, U[:, i])
        line_v.set_data(x, V[:, i])
        return line_u, line_v

    ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), init_func=init,interval=100, blit=True)
    return ani

def save_snapshots():
    times = [1,5,10,15,20,25]
    indices = [np.argmin(np.abs(t_eval - t)) for t in times]

    filename_wo_ext = f"{initial_condition}_K2_{K2}_dv_{dv:.2f}"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "../result/animation", filename_wo_ext)
    os.makedirs(base_dir, exist_ok=True)

    for i, t_idx in enumerate(indices):
        plt.figure(figsize=(10, 5))
        plt.plot(x, U[:, t_idx], label='u (slow)', linewidth=2.5)
        plt.plot(x, V[:, t_idx], label='v (fast)', linewidth=1.5)
        plt.xlabel("Space (x)")
        plt.ylabel("Population")
        plt.title(f"Population at t = {int(t_eval[t_idx])}")
        plt.legend()
        plt.grid(True)

        snapshot_path = os.path.join(base_dir, f"snapshot_t{int(t_eval[t_idx])}.png")
        plt.savefig(snapshot_path)
        plt.close()
        print(f"Saved snapshot: {snapshot_path}")


def show_animation():
    ani = create_animation()
    plt.show()

def save_animation():
    ani = create_animation()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result")
    result_dir = os.path.join(result_dir, "animation")
    os.makedirs(result_dir, exist_ok=True)

    filename = f"{initial_condition}_K2_{K2}_dv_{dv:.2f}.gif"
    filepath = os.path.join(result_dir, filename)

    ani.save(filepath, writer=PillowWriter(fps=10))
    print(f"Animation saved to {filepath}")

save_animation()
save_snapshots()