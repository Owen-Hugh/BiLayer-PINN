import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_predictions(model, domainGeom, numPtsU=10, numPtsV=10, output_dim=0):
    """
    Plots the model predictions over the 2D spatiotemporal domain.
    output_dim specifies which output dimension (0 or 1) to plot.
    """
    # Generate a grid of points over the domain
    xPhys, tPhys = domainGeom.getUnifIntPts(numPtsU, numPtsV, [1,1,1,1])
    X_plot = np.concatenate((xPhys, tPhys), axis=1).astype(np.float32)
    X_plot = torch.from_numpy(X_plot).to(next(model.parameters()).device)

    # Make predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        u_pred = model(X_plot).cpu().numpy()

    # Reshape predictions for plotting
    x_vals = xPhys[:, 0].reshape((numPtsU, numPtsV))
    t_vals = tPhys[:, 0].reshape((numPtsU, numPtsV))
    u_vals = u_pred[:, output_dim].reshape((numPtsU, numPtsV))  # 选择第 output_dim 维

    # Create a contour plot
    plt.figure(figsize=(10, 10))
    plt.contourf(x_vals, t_vals, u_vals, 255, cmap=plt.cm.jet)
    plt.colorbar(label=f'u(x, t) - output dim {output_dim}')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'PINN Model Predictions (output_dim={output_dim})')
    plt.show()

def plot_combined_predictions(model_a, model_b, domainGeom_a, domainGeom_b, numPtsU=10, numPtsV=10, output_dim=0):
    """
    Plots the combined predictions of two models over the spatiotemporal domain.
    output_dim specifies which output dimension (0 or 1) to plot.
    """
    # Generate a grid of points over the domain for material A
    xPhys_a, tPhys_a = domainGeom_a.getUnifIntPts(numPtsU, numPtsV, [1, 1, 1, 1])
    X_plot_a = np.concatenate((xPhys_a, tPhys_a), axis=1).astype(np.float32)
    X_plot_a = torch.from_numpy(X_plot_a).to(next(model_a.parameters()).device)

    # Generate a grid of points over the domain for material B
    xPhys_b, tPhys_b = domainGeom_b.getUnifIntPts(numPtsU, numPtsV, [1, 1, 1, 1])
    X_plot_b = np.concatenate((xPhys_b, tPhys_b), axis=1).astype(np.float32)
    X_plot_b = torch.from_numpy(X_plot_b).to(next(model_b.parameters()).device)

    # Make predictions for both models
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        u_pred_a = model_a(X_plot_a).cpu().numpy()
        u_pred_b = model_b(X_plot_b).cpu().numpy()

    # Reshape predictions for plotting
    x_vals_a = xPhys_a[:, 0].reshape((numPtsU, numPtsV))
    t_vals_a = tPhys_a[:, 0].reshape((numPtsU, numPtsV))
    u_vals_a = u_pred_a[:, output_dim].reshape((numPtsU, numPtsV))

    x_vals_b = xPhys_b[:, 0].reshape((numPtsU, numPtsV))
    t_vals_b = tPhys_b[:, 0].reshape((numPtsU, numPtsV))
    u_vals_b = u_pred_b[:, output_dim].reshape((numPtsU, numPtsV))

    # Create a combined grid for plotting
    x_vals_combined = np.concatenate((x_vals_a, x_vals_b), axis=0)
    t_vals_combined = np.concatenate((t_vals_a, t_vals_b), axis=0)
    u_vals_combined = np.concatenate((u_vals_a, u_vals_b), axis=0)

    # Create a contour plot for the combined data
    plt.figure(figsize=(10, 10))
    plt.contourf(x_vals_combined, t_vals_combined, u_vals_combined, 255, cmap=plt.cm.jet)
    plt.colorbar(label=f'u(x, t) - output dim {output_dim}')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Combined PINN Model Predictions (output_dim={output_dim})')
    plt.show()


def plot_predictions_double(model, domainGeom, numPtsU=10, numPtsV=10, output_dim=0):
    """
    Plots the model predictions over the 2D spatiotemporal domain.
    output_dim specifies which output dimension (0 or 1) to plot.
    """
    # Generate a grid of points over the domain
    xPhys, tPhys = domainGeom.getUnifIntPts(numPtsU, numPtsV, [1,1,1,1])
    X_plot = np.concatenate((xPhys, tPhys), axis=1).astype(np.float64)
    X_plot = torch.from_numpy(X_plot).double().to(next(model.parameters()).device)

    # Make predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        u_pred = model(X_plot).cpu().numpy()

    # Reshape predictions for plotting
    x_vals = xPhys[:, 0].reshape((numPtsU, numPtsV))
    t_vals = tPhys[:, 0].reshape((numPtsU, numPtsV))
    u_vals = u_pred[:, output_dim].reshape((numPtsU, numPtsV))  # 选择第 output_dim 维

    # Create a contour plot
    plt.figure(figsize=(10, 10))
    plt.contourf(x_vals, t_vals, u_vals, 255, cmap=plt.cm.jet)
    plt.colorbar(label=f'u(x, t) - output dim {output_dim}')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'PINN Model Predictions (output_dim={output_dim})')
    plt.show()

def plot_combined_predictions_double(model_a, model_b, domainGeom_a, domainGeom_b, numPtsU=10, numPtsV=10, output_dim=0):
    """
    Plots the combined predictions of two models over the spatiotemporal domain.
    output_dim specifies which output dimension (0 or 1) to plot.
    """
    # Generate a grid of points over the domain for material A
    xPhys_a, tPhys_a = domainGeom_a.getUnifIntPts(numPtsU, numPtsV, [1, 1, 1, 1])
    X_plot_a = np.concatenate((xPhys_a, tPhys_a), axis=1).astype(np.float64)
    X_plot_a = torch.from_numpy(X_plot_a).double().to(next(model_a.parameters()).device)

    # Generate a grid of points over the domain for material B
    xPhys_b, tPhys_b = domainGeom_b.getUnifIntPts(numPtsU, numPtsV, [1, 1, 1, 1])
    X_plot_b = np.concatenate((xPhys_b, tPhys_b), axis=1).astype(np.float64)
    X_plot_b = torch.from_numpy(X_plot_b).double().to(next(model_b.parameters()).device)

    # Make predictions for both models
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        u_pred_a = model_a(X_plot_a).cpu().numpy()
        u_pred_b = model_b(X_plot_b).cpu().numpy()

    # Reshape predictions for plotting
    x_vals_a = xPhys_a[:, 0].reshape((numPtsU, numPtsV))
    t_vals_a = tPhys_a[:, 0].reshape((numPtsU, numPtsV))
    u_vals_a = u_pred_a[:, output_dim].reshape((numPtsU, numPtsV))

    x_vals_b = xPhys_b[:, 0].reshape((numPtsU, numPtsV))
    t_vals_b = tPhys_b[:, 0].reshape((numPtsU, numPtsV))
    u_vals_b = u_pred_b[:, output_dim].reshape((numPtsU, numPtsV))

    # Create a combined grid for plotting
    x_vals_combined = np.concatenate((x_vals_a, x_vals_b), axis=0)
    t_vals_combined = np.concatenate((t_vals_a, t_vals_b), axis=0)
    u_vals_combined = np.concatenate((u_vals_a, u_vals_b), axis=0)

    # Create a contour plot for the combined data
    plt.figure(figsize=(10, 10))
    plt.contourf(x_vals_combined, t_vals_combined, u_vals_combined, 255, cmap=plt.cm.jet)
    plt.colorbar(label=f'u(x, t) - output dim {output_dim}')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Combined PINN Model Predictions (output_dim={output_dim})')
    plt.show()