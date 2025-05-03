import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import matplotlib.gridspec as gridspec
import sys
from network import *

plt.rcParams.update({
    "text.usetex": True,   
    "font.family": "serif",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "figure.figsize": (16, 8),
})

base_path = "FEM/moving_interface_complex/"
file_path_c = os.path.join(base_path, "moving_interface_long_c.csv")
file_path_u = os.path.join(base_path, "moving_interface_long_u.csv")

c_data = pd.read_csv(file_path_c)  # columns: time, x, value
u_data = pd.read_csv(file_path_u)  # columns: time, x, value

c_data.sort_values(["time", "x"], inplace=True, ignore_index=True)
u_data.sort_values(["time", "x"], inplace=True, ignore_index=True)

model_base_paths = [
    "model/moving_interface_complex/XPINN/",
    "model/moving_interface_complex/XDEM/",
    "model/moving_interface_complex/AXPINN/",
    "model/moving_interface_complex/AXDEM/"
]
model_names = ["XPINN", "XDEM", "AX-PINN", "AX-DEM"] 

def load_model_set(model_path, device="cpu"):
    diff_a = torch.load(os.path.join(model_path, "model_diffusion_a.pth"), map_location=device)
    diff_b = torch.load(os.path.join(model_path, "model_diffusion_b.pth"), map_location=device)
    phys_a = torch.load(os.path.join(model_path, "model_physics_a.pth"),  map_location=device)
    phys_b = torch.load(os.path.join(model_path, "model_physics_b.pth"),  map_location=device)
    
    diff_a.eval()
    diff_b.eval()
    phys_a.eval()
    phys_b.eval()
    
    return {
        "diff_a": diff_a,
        "diff_b": diff_b,
        "phys_a": phys_a,
        "phys_b": phys_b
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_sets = []
for path in model_base_paths:
    model_dict = load_model_set(path, device=device)
    model_sets.append(model_dict)

time_slices = [0.25, 0.50, 0.75, 1.00]
def get_comsol_slice(dataframe, t_target, tol=1e-8):
    sub = dataframe[np.isclose(dataframe["time"], t_target, atol=tol)]
    sub = sub.sort_values("x")
    return sub["x"].values, sub["value"].values

def predict_c_u(model_set, x_array, t_scalar):
    interface_t = 0.9 - (t_scalar ** 0.5) * 0.3

    c_pred = np.zeros_like(x_array)
    u_pred = np.zeros_like(x_array)
    
    left_mask = (x_array <= interface_t)
    right_mask= ~left_mask 
    
    x_left = x_array[left_mask]
    t_left = np.full_like(x_left, t_scalar)
    if len(x_left)>0:
        inp_left = np.stack([x_left, t_left], axis=-1)
        inp_left_tensor = torch.tensor(inp_left, dtype=torch.float32, device=device)
        with torch.no_grad():
            c_left = model_set["diff_a"](inp_left_tensor).cpu().numpy().flatten()
            u_left = model_set["phys_a"](inp_left_tensor).cpu().numpy().flatten()
        c_pred[left_mask] = c_left
        u_pred[left_mask] = u_left
    
    # right
    x_right= x_array[right_mask]
    t_right= np.full_like(x_right, t_scalar)
    if len(x_right)>0:
        inp_right = np.stack([x_right, t_right], axis=-1)
        inp_right_tensor = torch.tensor(inp_right, dtype=torch.float32, device=device)
        with torch.no_grad():
            c_right = model_set["diff_b"](inp_right_tensor).cpu().numpy().flatten()
            u_right = model_set["phys_b"](inp_right_tensor).cpu().numpy().flatten()
        c_pred[right_mask] = c_right
        u_pred[right_mask] = u_right
    
    return c_pred, u_pred

fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.4)

labels_grid = [
    ["(a)", "(b)", "(c)", "(d)"],
    ["(e)", "(f)", "(g)", "(h)"]
]

x_limits = [0.5, 1.0]
y_limits_c = [0, 3.0]
y_limits_u = [0, 0.8]
x_label = r"$x$"
c_label = r"$C(x,t)$"
u_label = r"$u(x,t)$"

for col_i, t_val in enumerate(time_slices):
    x_c, c_comsol = get_comsol_slice(c_data, t_val)
    x_u, u_comsol = get_comsol_slice(u_data, t_val)
    
    ax_c = plt.subplot(gs[0, col_i])
    ax_c.plot(x_c, c_comsol, 'b-', linewidth=3, label="FEM")
    
    for model_dict, model_name in zip(model_sets, model_names):
        c_pred, _ = predict_c_u(model_dict, x_c, t_val)
        ax_c.plot(x_c, c_pred, '--', linewidth=2, label=model_name)
    
    ax_c.set_title(f"{labels_grid[0][col_i]}  $t = {t_val:.2f}$", fontsize=18)
    ax_c.set_xlabel(x_label, fontsize=18)
    ax_c.set_ylabel(c_label, fontsize=18)
    ax_c.set_xlim(x_limits)
    ax_c.set_ylim(y_limits_c)
    ax_c.grid(True, linestyle="--", alpha=0.6)
    ax_c.legend(loc="best", fontsize=11)
    
    ax_u = plt.subplot(gs[1, col_i])
    ax_u.plot(x_u, u_comsol, 'b-', linewidth=3, label="FEM")
    
    for model_dict, model_name in zip(model_sets, model_names):
        _, u_pred = predict_c_u(model_dict, x_u, t_val)
        ax_u.plot(x_u, u_pred, '--', linewidth=2, label=model_name)
    
    ax_u.set_title(f"{labels_grid[1][col_i]}  $t = {t_val:.2f}$", fontsize=18)
    ax_u.set_xlabel(x_label, fontsize=18)
    ax_u.set_ylabel(u_label, fontsize=18)
    ax_u.set_xlim(x_limits)
    ax_u.set_ylim(y_limits_u)
    ax_u.grid(True, linestyle="--", alpha=0.6)
    ax_u.legend(loc="best", fontsize=11)

save_dir = "figure/moving_interface_complex"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir, "comsol_model_moving_interface_comparison.svg"), 
            format="svg", bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "comsol_model_moving_interface_comparison.pdf"), 
            format="pdf", bbox_inches="tight")
plt.show()
