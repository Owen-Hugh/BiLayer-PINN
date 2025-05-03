import pandas as pd
import numpy as np
import torch

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

    # Calculate L^2 error between COMSOL and deep learning predictions
    U_error_diffusion = np.linalg.norm(U_comsol_c - U_pred_diffusion) / np.linalg.norm(U_comsol_c)
    U_error_physics = np.linalg.norm(U_comsol_u - U_pred_physics) / np.linalg.norm(U_comsol_u)

    print(f'L^2 errors saved for {model_name}:')
    print(f'  Diffusion L^2 error: {U_error_diffusion}')
    print(f'  Physics L^2 error: {U_error_physics}')
