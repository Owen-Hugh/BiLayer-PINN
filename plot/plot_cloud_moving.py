import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import os
from scipy.interpolate import griddata

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

base_path = 'FEM/moving_interface_complex/'
file_path_c = os.path.join(base_path, 'moving_interface_long_c.csv')
file_path_u = os.path.join(base_path, 'moving_interface_long_u.csv')

df_c = pd.read_csv(file_path_c)  # columns: [time, x, value]
df_u = pd.read_csv(file_path_u)  # columns: [time, x, value]
df_c.sort_values(['time','x'], inplace=True, ignore_index=True)
df_u.sort_values(['time','x'], inplace=True, ignore_index=True)

t_min, t_max = df_c["time"].min(), df_c["time"].max()
x_min, x_max = df_c["x"].min(),    df_c["x"].max()
print(f"Time range: [{t_min}, {t_max}], X range: [{x_min}, {x_max}]")

Nt, Nx = 200, 200
t_lin = np.linspace(t_min, t_max, Nt)
x_lin = np.linspace(x_min, x_max, Nx)
T_grid, X_grid = np.meshgrid(t_lin, x_lin, indexing='ij') 

points_c = (df_c["time"].values, df_c["x"].values)  # shape(N,2)
values_c = df_c["value"].values                    # shape(N,)
c_comsol_2d = griddata(points_c, values_c,
                       (T_grid.ravel(), X_grid.ravel()),
                       method='linear')
c_comsol_2d = c_comsol_2d.reshape(T_grid.shape) 

points_u = (df_u["time"].values, df_u["x"].values)
values_u = df_u["value"].values
u_comsol_2d = griddata(points_u, values_u,
                       (T_grid.ravel(), X_grid.ravel()),
                       method='linear')
u_comsol_2d = u_comsol_2d.reshape(T_grid.shape) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_point(x_val, t_val, model_diff_a, model_diff_b, model_phys_a, model_phys_b):
    interface_t = 0.9 - (t_val ** 0.5) * 0.3  # or (1.0 + 0.5*t_val)
    inp = torch.tensor([[x_val, t_val]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        if x_val <= interface_t:
            c_out = model_diff_a(inp).cpu().numpy().item()
            u_out = model_phys_a(inp).cpu().numpy().item()
        else:
            c_out = model_diff_b(inp).cpu().numpy().item()
            u_out = model_phys_b(inp).cpu().numpy().item()
    return c_out, u_out


model_base_paths = [
    "model/moving_interface_complex/XPINN/",
    "model/moving_interface_complex/XDEM/",
    "model/moving_interface_complex/AXPINN/",
    "model/moving_interface_complex/AXDEM/"
]
titles = ["XPINN", "XDEM", "AX-PINN", "AX-DEM"] 
model_diffusions = []
model_physicses = []

for path in model_base_paths:
    m_diff_a = torch.load(os.path.join(path, "model_diffusion_a.pth"), map_location=device)
    m_diff_b = torch.load(os.path.join(path, "model_diffusion_b.pth"), map_location=device)
    m_phys_a = torch.load(os.path.join(path, "model_physics_a.pth"),  map_location=device)
    m_phys_b = torch.load(os.path.join(path, "model_physics_b.pth"),  map_location=device)
    
    m_diff_a.eval(); m_diff_b.eval()
    m_phys_a.eval(); m_phys_b.eval()
    
    model_diffusions.append((m_diff_a, m_diff_b))
    model_physicses.append((m_phys_a, m_phys_b))

U_preds_diffusion = []
U_preds_physics   = []
U_errors_diffusion= []
U_errors_physics  = []

N_total = Nt*Nx 
for i, (diff_pair, phys_pair) in enumerate(zip(model_diffusions, model_physicses)):
    diff_a, diff_b = diff_pair
    phys_a, phys_b = phys_pair
    
    c_pred_2d = np.zeros_like(c_comsol_2d)
    u_pred_2d = np.zeros_like(u_comsol_2d)
    
    for idx_t in range(Nt):
        for idx_x in range(Nx):
            xval = X_grid[idx_t, idx_x]
            tval = T_grid[idx_t, idx_x]
            c_val, u_val = predict_point(xval, tval, diff_a, diff_b, phys_a, phys_b)
            c_pred_2d[idx_t, idx_x] = c_val
            u_pred_2d[idx_t, idx_x] = u_val

    c_error_2d = np.abs(c_comsol_2d - c_pred_2d)
    u_error_2d = np.abs(u_comsol_2d - u_pred_2d)
    
    U_preds_diffusion.append(c_pred_2d)
    U_preds_physics.append(u_pred_2d)
    U_errors_diffusion.append(c_error_2d)
    U_errors_physics.append(u_error_2d)

fig1, axes1 = plt.subplots(1, 2, figsize=(12,6))

x_min, x_max = x_lin.min(), x_lin.max()
t_min, t_max = t_lin.min(), t_lin.max()

# (a) c_comsol
ax_c = axes1[0]
img_c = ax_c.imshow(c_comsol_2d, interpolation='nearest', cmap='jet',
                    extent=[x_min, x_max, t_min, t_max],
                    origin='lower', aspect='auto')
divider = make_axes_locatable(ax_c)
cax_c = divider.append_axes("right", size="5%", pad=0.10)
plt.colorbar(img_c, cax=cax_c)
ax_c.set_xlabel(r"$x^*$", fontsize=16)
ax_c.set_ylabel(r"$t^*$", fontsize=16)
ax_c.set_title("(a) COMSOL $C^*(x,t)$", fontsize=16)

# (b) u_comsol
ax_u = axes1[1]
img_u = ax_u.imshow(u_comsol_2d, interpolation='nearest', cmap='jet',
                    extent=[x_min, x_max, t_min, t_max],
                    origin='lower', aspect='auto')
divider = make_axes_locatable(ax_u)
cax_u = divider.append_axes("right", size="5%", pad=0.10)
plt.colorbar(img_u, cax=cax_u)
ax_u.set_xlabel(r"$x^*$", fontsize=16)
ax_u.set_ylabel(r"$t^*$", fontsize=16)
ax_u.set_title("(b) COMSOL $u^*(x,t)$", fontsize=16)

plt.tight_layout()
os.makedirs("figure/moving_interface_complex", exist_ok=True)
plt.savefig("figure/moving_interface_complex/comsol_cloud.svg", format="svg", bbox_inches="tight")
plt.savefig("figure/moving_interface_complex/comsol_cloud.pdf", format="pdf", bbox_inches="tight")
plt.show()

fig2, axes2 = plt.subplots(4, 4, figsize=(24,22))
labels_grid = [
    ["(a)", "(b)", "(c)", "(d)"],
    ["(e)", "(f)", "(g)", "(h)"],
    ["(i)", "(j)", "(k)", "(l)"],
    ["(m)", "(n)", "(o)", "(p)"]
]

vmin_c = min([np.nanmin(U) for U in U_preds_diffusion])
vmax_c = max([np.nanmax(U) for U in U_preds_diffusion])

vmin_c_err = min([np.nanmin(U) for U in U_errors_diffusion])
vmax_c_err = max([np.nanmax(U) for U in U_errors_diffusion])

vmin_u = min([np.nanmin(U) for U in U_preds_physics])
vmax_u = max([np.nanmax(U) for U in U_preds_physics])

vmin_u_err = min([np.nanmin(U) for U in U_errors_physics])
vmax_u_err = max([np.nanmax(U) for U in U_errors_physics])

# 行1: C^*(x,t) predictions
for col_i in range(4):
    ax = axes2[0, col_i]
    c_pred_2d = U_preds_diffusion[col_i]
    h = ax.imshow(c_pred_2d, interpolation='nearest', cmap='jet',
                  extent=[x_min, x_max, t_min, t_max],
                  origin='lower', aspect='auto',
                  vmin=vmin_c, vmax=vmax_c)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig2.colorbar(h, cax=cax)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$t$", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title(f"{labels_grid[0][col_i]}  Prediction $C(x,t)$ - {titles[col_i]}",
                 fontsize=20)

# 行2: C^* error
for col_i in range(4):
    ax = axes2[1, col_i]
    c_err_2d = U_errors_diffusion[col_i]
    h = ax.imshow(c_err_2d, interpolation='nearest', cmap='jet',
                  extent=[x_min, x_max, t_min, t_max],
                  origin='lower', aspect='auto',
                  vmin=vmin_c_err, vmax=vmax_c_err)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig2.colorbar(h, cax=cax)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$t$", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title(f"{labels_grid[1][col_i]}  Error $C(x,t)$ - {titles[col_i]}",
                 fontsize=20)

# 行3: U^*(x,t) predictions
for col_i in range(4):
    ax = axes2[2, col_i]
    u_pred_2d = U_preds_physics[col_i]
    h = ax.imshow(u_pred_2d, interpolation='nearest', cmap='jet',
                  extent=[x_min, x_max, t_min, t_max],
                  origin='lower', aspect='auto',
                  vmin=vmin_u, vmax=vmax_u)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig2.colorbar(h, cax=cax)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$t$", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title(f"{labels_grid[2][col_i]}  Prediction $u(x,t)$ - {titles[col_i]}",
                 fontsize=20)

# 行4: U^* error
for col_i in range(4):
    ax = axes2[3, col_i]
    u_err_2d = U_errors_physics[col_i]
    h = ax.imshow(u_err_2d, interpolation='nearest', cmap='jet',
                  extent=[x_min, x_max, t_min, t_max],
                  origin='lower', aspect='auto',
                  vmin=vmin_u_err, vmax=vmax_u_err)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig2.colorbar(h, cax=cax)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$t$", fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_title(f"{labels_grid[3][col_i]}  Error $u(x,t)$ - {titles[col_i]}",
                 fontsize=20)

plt.tight_layout()
plt.savefig("figure/moving_interface_complex/deep_learning_comparison.svg", format="svg", bbox_inches="tight")
plt.savefig("figure/moving_interface_complex/deep_learning_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()
