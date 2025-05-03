import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import os

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
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
base_path = 'FEM/softening_modulus/'
model_paths = ['model/softening_modulus/XPINN/', 'model/softening_modulus/XDEM/', 'model/softening_modulus/AXPINN/', 'model/softening_modulus/AXDEM/']
titles = ['X-PINN', 'X-DEM', 'AX-PINN', 'AX-DEM']

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

# Prepare the data for plotting
X, T = np.meshgrid(x_points, t_points)

# Load models and perform predictions for XPINN, IPINN, EIPINN
U_preds_diffusion = []
U_preds_physics = []
U_errors_diffusion = []
U_errors_physics = []

for model_path in model_paths:
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

    # Calculate error between COMSOL and deep learning predictions
    U_error_diffusion = np.abs(U_comsol_c - U_pred_diffusion)
    U_error_physics = np.abs(U_comsol_u - U_pred_physics)

    # Store predictions and errors
    U_preds_diffusion.append(U_pred_diffusion)
    U_preds_physics.append(U_pred_physics)
    U_errors_diffusion.append(U_error_diffusion)
    U_errors_physics.append(U_error_physics)

# Define figure size for square-shaped subplots
figsize = (12, 6)

# Plotting COMSOL data
fig1, axes1 = plt.subplots(1, 2, figsize=figsize)

# Plot the COMSOL data for C
ax = axes1[0]
h = ax.imshow(U_comsol_c, interpolation='nearest', cmap='jet',
              extent=[x_points.min(), x_points.max(), t_points.min(), t_points.max()],
              origin='lower', aspect='auto')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig1.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.set_xlabel(r'${x}^*$', size=20)
ax.set_ylabel(r'${t}^*$', size=20)
ax.set_title("(a) COMSOL Simulation ${C}^*(x,t)$", fontsize=20, loc='left')
ax.tick_params(labelsize=15)

# Plot the COMSOL data for U
ax = axes1[1]
h = ax.imshow(U_comsol_u, interpolation='nearest', cmap='jet',
              extent=[x_points.min(), x_points.max(), t_points.min(), t_points.max()],
              origin='lower', aspect='auto')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig1.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.set_xlabel(r'${x}^*$', size=20)
ax.set_ylabel(r'${t}^*$', size=20)
ax.set_title("(b) COMSOL Simulation ${u}^*(x,t)$", fontsize=20, loc='left')
ax.tick_params(labelsize=15)

plt.tight_layout()
# Save the plot
plt.savefig('figure/softening_modulus/comsol_cloud_latex.svg', format='svg')
plt.savefig('figure/softening_modulus/comsol_cloud_latex.pdf', format='pdf')
plt.show()


# 设置 colorbar 统一上下限
vmin_c = min([np.nanmin(U) for U in U_preds_diffusion])
vmax_c = max([np.nanmax(U) for U in U_preds_diffusion])

vmin_c_err = min([np.nanmin(U) for U in U_errors_diffusion])
vmax_c_err = max([np.nanmax(U) for U in U_errors_diffusion])

vmin_u = min([np.nanmin(U) for U in U_preds_physics])
vmax_u = max([np.nanmax(U) for U in U_preds_physics])

vmin_u_err = min([np.nanmin(U) for U in U_errors_physics])
vmax_u_err = max([np.nanmax(U) for U in U_errors_physics])

# Plotting deep learning predictions and errors
fig2, axes2 = plt.subplots(4, 4, figsize=(24, 22))

# Adjust aspect ratio to ensure square subplots
for ax in axes2.flat:
    ax.set_aspect('auto')
    
labels_grid = [
    ["(a)", "(b)", "(c)", "(d)"],
    ["(e)", "(f)", "(g)", "(h)"],
    ["(i)", "(j)", "(k)", "(l)"],
    ["(m)", "(n)", "(o)", "(p)"]
]

# Plot the deep learning prediction for C
for i in range(4):
    ax = axes2[0, i]
    h = ax.imshow(U_preds_diffusion[i], interpolation='nearest', cmap='jet',
                  extent=[x_points.min(), x_points.max(), t_points.min(), t_points.max()],
                  origin='lower', aspect='auto',
                  vmin=vmin_c, vmax=vmax_c)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig2.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'${x}$', size=20)
    ax.set_ylabel(r'${t}$', size=20)
    ax.set_title(f"{labels_grid[0][i]} Prediction $C(x,t)$ - {titles[i]}",
                 fontsize=20)
    ax.tick_params(labelsize=15)

# Plot the error for C
for i in range(4):
    ax = axes2[1, i]
    h = ax.imshow(U_errors_diffusion[i], interpolation='nearest', cmap='jet',
                  extent=[x_points.min(), x_points.max(), t_points.min(), t_points.max()],
                  origin='lower', aspect='auto',
                  vmin=vmin_c_err, vmax=vmax_c_err)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig2.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'${x}$', size=20)
    ax.set_ylabel(r'${t}$', size=20)
    ax.set_title(f"{labels_grid[1][i]} Error $C(x,t)$ - {titles[i]}",
                 fontsize=20)
    ax.tick_params(labelsize=15)

# Plot the deep learning prediction for U
for i in range(4):
    ax = axes2[2, i]
    h = ax.imshow(U_preds_physics[i], interpolation='nearest', cmap='jet',
                  extent=[x_points.min(), x_points.max(), t_points.min(), t_points.max()],
                  origin='lower', aspect='auto',
                  vmin=vmin_u, vmax=vmax_u)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig2.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'${x}$', size=20)
    ax.set_ylabel(r'${t}$', size=20)
    ax.set_title(f"{labels_grid[2][i]} Prediction $u(x,t)$ - {titles[i]}",
                 fontsize=20)
    ax.tick_params(labelsize=15)

# Plot the error for U
for i in range(4):
    ax = axes2[3, i]
    h = ax.imshow(U_errors_physics[i], interpolation='nearest', cmap='jet',
                  extent=[x_points.min(), x_points.max(), t_points.min(), t_points.max()],
                  origin='lower', aspect='auto',
                  vmin=vmin_u_err, vmax=vmax_u_err)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig2.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'${x}$', size=20)
    ax.set_ylabel(r'${t}$', size=20)
    ax.set_title(f"{labels_grid[3][i]} Error $u(x,t)$ - {titles[i]}",
                 fontsize=20)
    ax.tick_params(labelsize=15)

plt.tight_layout()
# Save the plot
plt.savefig('figure/softening_modulus/deep_learning_comparison.svg', format='svg')
plt.savefig('figure/softening_modulus/deep_learning_comparison.pdf', format='pdf')
plt.show()