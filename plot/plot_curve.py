import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import matplotlib.gridspec as gridspec
import sys
from network import Network

plt.rcParams.update({
    "text.usetex": True,   # Use LaTeX for text rendering
    "font.family": "serif",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "figure.figsize": (16, 8),
})

# Define base paths for data and model files
base_path = 'FEM/small_deformation/'
model_paths = ['model/small_deformation/XPINN/', 'model/small_deformation/XDEM/', 'model/small_deformation/AXPINN/', 'model/small_deformation/AXDEM/']
model_names = ['X-PINN', 'X-DEM', 'AX-PINN', 'AX-DEM']

# Load the reshaped COMSOL data
file_path_c = f'{base_path}comsol_data_c.csv'
file_path_u = f'{base_path}comsol_data_u.csv'
reshaped_data_c = pd.read_csv(file_path_c, header=None)
reshaped_data_u = pd.read_csv(file_path_u, header=None)

# Define spatial and temporal domains
x_points = np.linspace(0.5, 1.0, reshaped_data_c.shape[1])  # x-coordinates
t_points = np.linspace(0, 1.0, reshaped_data_c.shape[0])    # t-coordinates

# Reverse the order of time values to match expected orientation
U_comsol_c = reshaped_data_c.iloc[::-1].values
U_comsol_u = reshaped_data_u.iloc[::-1].values

# Prepare the data for predictions
X, T = np.meshgrid(x_points, t_points)

# Load models and perform predictions for XPINN, IPINN, EIPINN
U_preds_diffusion = []
U_preds_physics = []

for model_path, model_name in zip(model_paths, model_names):
    # Select GPU or CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load models
    model_diffusion_a = torch.load(f'{model_path}model_diffusion_a.pth', map_location=device)
    model_diffusion_b = torch.load(f'{model_path}model_diffusion_b.pth', map_location=device)
    model_physics_a = torch.load(f'{model_path}model_physics_a.pth', map_location=device)
    model_physics_b = torch.load(f'{model_path}model_physics_b.pth', map_location=device)

    model_diffusion_a.eval()  # Set model to evaluation mode
    model_diffusion_b.eval()  # Set model to evaluation mode
    model_physics_a.eval()  # Set model to evaluation mode
    model_physics_b.eval()  # Set model to evaluation mode

    # Deep learning model predictions
    interface = 0.80
    inputs_a = np.stack([X[X <= interface], T[X <= interface]], axis=-1)
    inputs_b = np.stack([X[X > interface], T[X > interface]], axis=-1)

    inputs_a_tensor = torch.tensor(inputs_a, dtype=torch.float32).to(device)
    inputs_b_tensor = torch.tensor(inputs_b, dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred_diffusion_a = model_diffusion_a(inputs_a_tensor).cpu().numpy()
        u_pred_diffusion_b = model_diffusion_b(inputs_b_tensor).cpu().numpy()
        u_pred_physics_a = model_physics_a(inputs_a_tensor).cpu().numpy()
        u_pred_physics_b = model_physics_b(inputs_b_tensor).cpu().numpy()

    # Initialize empty arrays for predictions
    U_pred_diffusion = np.full(X.shape, np.nan)
    U_pred_physics = np.full(X.shape, np.nan)

    # Assign predictions to the correct locations
    U_pred_diffusion[X <= interface] = u_pred_diffusion_a.flatten()
    U_pred_diffusion[X > interface] = u_pred_diffusion_b.flatten()
    U_pred_physics[X <= interface] = u_pred_physics_a.flatten()
    U_pred_physics[X > interface] = u_pred_physics_b.flatten()

    # Store predictions
    U_preds_diffusion.append(U_pred_diffusion)
    U_preds_physics.append(U_pred_physics)

# Define time slices to plot
time_slices = [0.25, 0.5, 0.75, 1.0]
time_indices = [int(t * (len(t_points) - 1)) for t in time_slices]

# Plotting the results
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 4)
gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.4, hspace=0.4)

# Define y-axis limits for consistency
x_limits = [x_points.min(), x_points.max()]
y_limits_c = [0, 4]
y_limits_u = [0.0, 7]

labels_grid = [
    ["(a)", "(b)", "(c)", "(d)"],
    ["(e)", "(f)", "(g)", "(h)"]
]

# First row: C plots at different time slices
for i, t_idx in enumerate(time_indices):
    ax = plt.subplot(gs[0, i])
    ax.plot(x_points, U_comsol_c[t_idx, :], 'b-', linewidth=3, label='FEM')
    for j, model_name in enumerate(model_names):
        ax.plot(x_points, U_preds_diffusion[j][t_idx, :], '--', linewidth=3, label=model_name)
    
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$C(x,t)$", fontsize=18)
    ax.set_title(f"{labels_grid[0][i]} $t = %.2f$" % time_slices[i], fontsize=18)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits_c)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

# Second row: U plots at different time slices
for i, t_idx in enumerate(time_indices):
    ax = plt.subplot(gs[1, i])
    ax.plot(x_points, U_comsol_u[t_idx, :], 'b-', linewidth=3, label='FEM')
    for j, model_name in enumerate(model_names):
        ax.plot(x_points, U_preds_physics[j][t_idx, :], '--', linewidth=3, label=model_name)
    
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$u(x,t)$", fontsize=18)
    ax.set_title(f"{labels_grid[1][i]} $t = %.2f$" % time_slices[i], fontsize=18)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits_u)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

# fig.legend(['COMSOL', 'XPINN', 'IPINN', 'EIPINN'], loc='upper center', bbox_to_anchor=(-0.8, -0.15), ncol=5, frameon=False, prop={'size': 15})
# Save the plot to the figure folder
save_dir = "figure/small_deformation"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir, "comsol_model_comparison_plot.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(save_dir, "comsol_model_comparison_plot.svg"), format="svg", bbox_inches="tight")

plt.show()