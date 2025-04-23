import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm

# model parameters
K1 = 500
K2_list = [250, 500, 1000]
du = 0.01
dv_list = [round(0.01 + 0.02 * i, 3) for i in range(10)]
t_span = (0, 150)
t_eval = np.linspace(t_span[0], t_span[1], 300)
initial_conditions = {
    "same_patch": [3, 0, 3, 0],
    "separate_patch_A": [3, 0, 0, 3],
    "separate_patch_B": [0, 3, 3, 0]
}

# Define the system of ODEs
def competition_model(t, y, K2, dv):
    u1, u2, v1, v2 = y
    du1 = u1 * (1 - (u1 + v1) / K1) - du * (u1 - u2)
    du2 = u2 * (1 - (u2 + v2) / K2) - du * (u2 - u1)
    dv1 = v1 * (1 - (u1 + v1) / K1) - dv * (v1 - v2)
    dv2 = v2 * (1 - (u2 + v2) / K2) - dv * (v2 - v1)
    return [du1, du2, dv1, dv2]

# Run simulations for all combinations of K2, dv, and initial condition
results = {}

for K2 in K2_list:
    results[K2] = {}
    for dv in dv_list:
        results[K2][dv] = {}
        for label, initial in initial_conditions.items():
            sol = solve_ivp(lambda t, y: competition_model(t, y, K2, dv),
                            t_span, initial, t_eval=t_eval)
            results[K2][dv][label] = sol


def save_plot_result(K2_key, dv_key, label_key):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "../result")
    result_dir = os.path.join(result_dir,label_key)
    os.makedirs(result_dir, exist_ok=True)

    sol=results[K2_key][dv_key][label_key]

    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], linestyle='-', color='blue',linewidth=2.5, label="u1")
    plt.plot(sol.t, sol.y[1], linestyle='--', color='blue',linewidth=2.5, label="u2")
    plt.plot(sol.t, sol.y[2], linestyle='-', color='red',linewidth=1.5, label="v1")
    plt.plot(sol.t, sol.y[3], linestyle='--', color='red',linewidth=1.5, label="v2")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title(f"dv={dv_key} | K2={K2_key} | Init={label_key}")
    plt.legend()
    plt.grid(True)

    filename = f"{label_key}_K2_{K2_key}_dv_{dv_key:.2f}.png"
    filepath = os.path.join(result_dir, filename)
    plt.savefig(filepath)
    plt.close()

def save_all_plots():
    total_tasks = len(K2_list) * len(dv_list) * len(initial_conditions)
    with tqdm(total=total_tasks, desc="Saving plots") as pbar:
        for K2_key in K2_list:
            for dv_key in dv_list:
                for label_key in initial_conditions.keys():
                    save_plot_result(K2_key, dv_key, label_key)
                    pbar.update(1)

save_all_plots()



