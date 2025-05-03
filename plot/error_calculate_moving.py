import pandas as pd
import numpy as np
import torch
import os
from scipy.interpolate import griddata

base_path = 'FEM/moving_interface_complex/'
file_path_c = os.path.join(base_path, 'moving_interface_long_c.csv')  # columns=[time,x,value] for c
file_path_u = os.path.join(base_path, 'moving_interface_long_u.csv')  # columns=[time,x,value] for u

df_c = pd.read_csv(file_path_c)
df_u = pd.read_csv(file_path_u)

df_c.sort_values(['time','x'], inplace=True, ignore_index=True)
df_u.sort_values(['time','x'], inplace=True, ignore_index=True)

t_min, t_max = df_c['time'].min(), df_c['time'].max()
x_min, x_max = df_c['x'].min(),    df_c['x'].max()
print(f"COMSOL data range:\n  time in [{t_min},{t_max}], x in [{x_min},{x_max}]")

Nt, Nx = 200, 200
t_lin = np.linspace(t_min, t_max, Nt)
x_lin = np.linspace(x_min, x_max, Nx)
T_grid, X_grid = np.meshgrid(t_lin, x_lin, indexing='ij')  # shape=(Nt,Nx)

points_c  = (df_c['time'].values, df_c['x'].values)  # shape(N,2)
values_c  = df_c['value'].values                    # shape(N,)
c_comsol_1d = griddata(points_c, values_c,
                       (T_grid.ravel(), X_grid.ravel()),
                       method='linear')  
c_comsol_2d = c_comsol_1d.reshape(T_grid.shape)  

points_u  = (df_u['time'].values, df_u['x'].values)
values_u  = df_u['value'].values
u_comsol_1d = griddata(points_u, values_u,
                       (T_grid.ravel(), X_grid.ravel()),
                       method='linear')
u_comsol_2d = u_comsol_1d.reshape(T_grid.shape) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_point(x_val, t_val, diff_a, diff_b, phys_a, phys_b):
    interface_t = 0.9 - (t_val ** 0.5) * 0.5
    inp = torch.tensor([[x_val, t_val]], dtype=torch.float32, device=device)
    with torch.no_grad():
        if x_val <= interface_t:
            c_out = diff_a(inp).cpu().numpy().item()
            u_out = phys_a(inp).cpu().numpy().item()
        else:
            c_out = diff_b(inp).cpu().numpy().item()
            u_out = phys_b(inp).cpu().numpy().item()
    return c_out, u_out

model_paths = [
    "model/moving_interface_complex/XPINN/",
    "model/moving_interface_complex/XDEM/",
    "model/moving_interface_complex/AXPINN/",
    "model/moving_interface_complex/AXDEM/"
]
model_names = ["XPINN", "XDEM", "AX-PINN", "AX-DEM"]

for path, name in zip(model_paths, model_names):
    diff_a = torch.load(os.path.join(path, 'model_diffusion_a.pth'), map_location=device)
    diff_b = torch.load(os.path.join(path, 'model_diffusion_b.pth'), map_location=device)
    phys_a = torch.load(os.path.join(path, 'model_physics_a.pth'),  map_location=device)
    phys_b = torch.load(os.path.join(path, 'model_physics_b.pth'),  map_location=device)
    
    diff_a.eval(); diff_b.eval()
    phys_a.eval(); phys_b.eval()
    
    c_pred_2d = np.zeros_like(c_comsol_2d)
    u_pred_2d = np.zeros_like(u_comsol_2d)
    
    for i_t in range(Nt):
        for j_x in range(Nx):
            x_val = X_grid[i_t, j_x]
            t_val = T_grid[i_t, j_x]
            c_val, u_val = predict_point(x_val, t_val, diff_a, diff_b, phys_a, phys_b)
            c_pred_2d[i_t, j_x] = c_val
            u_pred_2d[i_t, j_x] = u_val
    
    c_mask = ~np.isnan(c_comsol_2d)
    u_mask = ~np.isnan(u_comsol_2d)
    
    num_c = np.linalg.norm(c_comsol_2d[c_mask])  # denominator
    diff_c = np.linalg.norm(c_comsol_2d[c_mask] - c_pred_2d[c_mask])
    error_diffusion = diff_c / num_c
    
    num_u = np.linalg.norm(u_comsol_2d[u_mask])
    diff_u = np.linalg.norm(u_comsol_2d[u_mask] - u_pred_2d[u_mask])
    error_physics = diff_u / num_u
    
    print(f"\nModel: {name}")
    print(f"  Diffusion L^2 error (c): {error_diffusion:.6f}")
    print(f"  Physics   L^2 error (u): {error_physics:.6f}")
