import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
random = 1234
torch.manual_seed(random)
torch.cuda.manual_seed(random)
torch.cuda.manual_seed_all(random)
np.random.seed(random)
from utility.network import Network
from utility.WeightedLoss import AutomaticWeightedLoss
from utility.yaml_load import load_config

def f(y):
    return 0.9 - (y ** 0.5) * 0.3

def gauss_legendre_1d(n):
    """
    Returns (x_i, w_i), i=1..n
    as the Gauss-Legendre nodes and weights for n points on [-1,1].
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def map_to_segment(a, b, xi):
    """
    Linearly map xi (∈[-1,1]) to [a,b].
    """
    return 0.5*(b-a)*xi + 0.5*(a+b)

def map_weights(a, b, w_ref):
    """
    Map the weight w_ref on the reference interval [-1,1] to [a,b], and get w_phys=(b-a)/2 * w_ref.
    """
    return 0.5*(b-a)*w_ref

def get_gauss_points_subdomain_A(f, x_min, x_max, y_min, y_max, n_x, n_y):
    """
    For subdomain A: x ∈ [x_min, f(y)], y ∈ [y_min,y_max].
    Do Gauss-Legendre with n_y points at [y_min,y_max],
    For each y_i, do n_x points Gauss-Legendre at [x_min, f(y_i)].
    Return (ptsA, wtsA).
    ptsA.shape=(N,2), wtsA.shape=(N,)
    """
    xi_y, w_y = gauss_legendre_1d(n_y)
    y_nodes   = map_to_segment(y_min, y_max, xi_y)
    w_y_phys  = map_weights(y_min, y_max, w_y)

    xi_x, w_x = gauss_legendre_1d(n_x)

    pointsA = []
    weightsA = []

    for i in range(n_y):
        yi   = y_nodes[i]
        wyi  = w_y_phys[i]
        x_hi = f(yi)   
        x_lo = x_min  
        if x_hi<=x_lo:
            continue
        x_mapped  = map_to_segment(x_lo, x_hi, xi_x)
        w_x_phys  = map_weights(x_lo, x_hi, w_x)

        for j in range(n_x):
            pointsA.append((x_mapped[j], yi))
            weightsA.append( wyi * w_x_phys[j] )

    return np.array(pointsA, dtype=np.float32), np.array(weightsA, dtype=np.float32)


def get_gauss_points_subdomain_B(f, x_min, x_max, y_min, y_max, n_x, n_y):
    """
    For subdomain B: x ∈ [f(y), x_max], y ∈ [y_min,y_max].
    Similar to get_gauss_points_subdomain_A.
    """
    xi_y, w_y = gauss_legendre_1d(n_y)
    y_nodes   = map_to_segment(y_min, y_max, xi_y)
    w_y_phys  = map_weights(y_min, y_max, w_y)

    xi_x, w_x = gauss_legendre_1d(n_x)

    pointsB = []
    weightsB = []

    for i in range(n_y):
        yi   = y_nodes[i]
        wyi  = w_y_phys[i]
        x_lo = f(yi)
        x_hi = x_max
        if x_lo>=x_hi:
            continue
        x_mapped  = map_to_segment(x_lo, x_hi, xi_x)
        w_x_phys  = map_weights(x_lo, x_hi, w_x)

        for j in range(n_x):
            pointsB.append((x_mapped[j], yi))
            weightsB.append( wyi * w_x_phys[j] )

    return np.array(pointsB, dtype=np.float32), np.array(weightsB, dtype=np.float32)

def line_points(start, end, n):
    """
    Divide the line segment start->end evenly into n segments and return (n+1,2).
    """
    x_s, y_s = start
    x_e, y_e = end
    xs = np.linspace(x_s, x_e, n+1)
    ys = np.linspace(y_s, y_e, n+1)
    return np.column_stack((xs, ys))


def line_points_y(f, y_start, y_end, n):
    """
    Divide y ∈ [y_start,y_end] into n segments evenly, and calculate x_i=f(y_i) for each y_i.
    Return (n+1,2).
    """
    ys = np.linspace(y_start, y_end, n+1)
    xs = f(ys)
    return np.column_stack((xs, ys))


def calc_lam_theta(E, v):
    """
        Given material parameters E, v, return lam and theta (commonly in linear elastic constitutive).
        lam = v*E / [(1+v)*(1-2v)]
        theta = E / (1+v)
    """
    lam = v * E / ((1 + v) * (1 - 2*v))
    theta = E / (1 + v)
    return lam, theta

def calc_stress(eps_xx, eps_yy, eps_zz, c, lam, theta, omiga, cycle_number):
    """
        Calculate 3 principal stress + average stress according to (eps_xx, eps_yy, eps_zz, c, lam, theta, omiga);
        When cycle_number == 1, you can choose whether to ignore (omiga*c)
    """
    if cycle_number == 1:
        omiga_c = 0.0
    else:
        omiga_c = omiga * c

    stress_xx = lam*(eps_xx + eps_yy + eps_zz - omiga_c) + theta*(eps_xx - 1./3.*omiga_c)
    stress_yy = lam*(eps_xx + eps_yy + eps_zz - omiga_c) + theta*(eps_yy - 1./3.*omiga_c)
    stress_zz = lam*(eps_xx + eps_yy + eps_zz - omiga_c) + theta*(eps_zz - 1./3.*omiga_c)
    sigman    = (stress_xx + stress_yy + stress_zz)/3.0
    return stress_xx, stress_yy, stress_zz, sigman

def calc_flux(dc_dx, c, dsigman_dx, D, omiga, Rg, T, cycle_number):
    """
        Calculate flux based on (dc_dx, c, dsigman_dx, D, omiga, Rg, T, cycle_number).
    """
    if cycle_number == 1:
        flux = -D * dc_dx
    else:
        flux = -D * (dc_dx - omiga*c/(Rg*T)*dsigman_dx)
    return flux

class Multi_Diffusion:
    def __init__(self):
        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cycle_number = 1
        
        # create the network
        self.model_diffusion_a = self._build_network(2, 1, 32, 3, act=torch.nn.Tanh).to(self.device)
        self.model_diffusion_b = self._build_network(2, 1, 32, 3, act=torch.nn.Tanh).to(self.device)
        self.model_physics_a   = self._build_network(2, 1, 32, 3, act=lambda: torch.nn.SiLU()).to(self.device)
        self.model_physics_b   = self._build_network(2, 1, 32, 3, act=lambda: torch.nn.SiLU()).to(self.device)
        
        x_min, x_max = 0.5, 1.0
        y_min, y_max = 0.0, 1.0

        n_x, n_y = 16, 16
        X_Int_A, W_Int_A = get_gauss_points_subdomain_A(f, x_min, x_max, y_min, y_max, n_x, n_y)
        X_Int_B, W_Int_B = get_gauss_points_subdomain_B(f, x_min, x_max, y_min, y_max, n_x, n_y)
        num_bc = 16
        bc_left = line_points((x_min,y_min), (x_min,y_max), num_bc)
        bc_right= line_points((x_max,y_min), (x_max,y_max), num_bc)
        interface_line = line_points_y(f, y_min, y_max, num_bc)
        x0 = f(y_min) 
        bc_bottom_left  = line_points((x_min,y_min), (x0,y_min), num_bc)
        bc_bottom_right = line_points((x0,y_min),    (x_max,y_min), num_bc)
        # in domain
        self.X_Int_A = torch.tensor(X_Int_A,          dtype=torch.float32).to(self.device)
        self.X_Int_B = torch.tensor(X_Int_B,          dtype=torch.float32).to(self.device)
        self.W_Int_A = torch.tensor(W_Int_A,          dtype=torch.float32).to(self.device)
        self.W_Int_B = torch.tensor(W_Int_B,          dtype=torch.float32).to(self.device)
        # boundary & Interface & Initial
        self.XBnd_Left = torch.tensor(bc_left,          dtype=torch.float32).to(self.device)
        self.XInf_A = torch.tensor(interface_line,          dtype=torch.float32).to(self.device)
        self.XInit_A = torch.tensor(bc_bottom_left,          dtype=torch.float32).to(self.device)
         
        self.XBnd_Right = torch.tensor(bc_right,          dtype=torch.float32).to(self.device)
        self.XInf_B = torch.tensor(interface_line,          dtype=torch.float32).to(self.device)
        self.XInit_B = torch.tensor(bc_bottom_right,          dtype=torch.float32).to(self.device)
        # Boundary and Initial value
        self.XBnd_Left_Val = torch.zeros(len(self.XBnd_Left)).unsqueeze(1)
        self.XBnd_Right_Val = torch.ones(len(self.XBnd_Right)).unsqueeze(1)
        self.XInit_A_Val = torch.zeros(len(self.XInit_A)).unsqueeze(1)
        self.XInit_B_Val = torch.zeros(len(self.XInit_B)).unsqueeze(1)
        
        # to GPU
        self.X_Int_A = self.X_Int_A.to(self.device)
        self.X_Int_B = self.X_Int_B.to(self.device)
        self.XBnd_Left = self.XBnd_Left.to(self.device)
        self.XBnd_Right = self.XBnd_Right.to(self.device)
        self.XInit_A = self.XInit_A.to(self.device)
        self.XInit_B = self.XInit_B.to(self.device)
        self.XInf_A = self.XInf_A.to(self.device)
        self.XInf_B = self.XInf_B.to(self.device)
        self.XBnd_Left_Val = self.XBnd_Left_Val.to(self.device)
        self.XBnd_Right_Val = self.XBnd_Right_Val.to(self.device)
        self.XInit_A_Val = self.XInit_A_Val.to(self.device)
        self.XInit_B_Val = self.XInit_B_Val.to(self.device)
        self.W_Int_A = self.W_Int_A.to(self.device)
        self.W_Int_B = self.W_Int_B.to(self.device)
        
        self.X_Int_A.requires_grad = True
        self.X_Int_B.requires_grad = True
        self.XBnd_Left.requires_grad = True
        self.XBnd_Right.requires_grad = True
        self.XInf_A.requires_grad = True
        self.XInf_B.requires_grad = True

        # loss
        self.criterion = torch.nn.MSELoss()
        self.iter_diffusion_a = 0
        self.iter_diffusion_b = 0
        self.iter_physics_a   = 0
        self.iter_physics_b   = 0
        
        self.awl_diffusion_a = AutomaticWeightedLoss(5)
        self.awl_diffusion_b = AutomaticWeightedLoss(5)
        self.awl_physics_a   = AutomaticWeightedLoss(5)
        self.awl_physics_b   = AutomaticWeightedLoss(5)
        
        # optimizer
        self.optimizer_diffusion_a = torch.optim.Adam([
            {'params': self.model_diffusion_a.parameters()},
            {'params': self.awl_diffusion_a.parameters(), 'weight_decay': 0}
        ])
        self.optimizer_diffusion_b = torch.optim.Adam([
            {'params': self.model_diffusion_b.parameters()},
            {'params': self.awl_diffusion_b.parameters(), 'weight_decay': 0}
        ])
        self.optimizer_physics_a = torch.optim.Adam([
            {'params': self.model_physics_a.parameters()},
            {'params': self.awl_physics_a.parameters(), 'weight_decay': 0}
        ])
        self.optimizer_physics_b = torch.optim.Adam([
            {'params': self.model_physics_b.parameters()},
            {'params': self.awl_physics_b.parameters(), 'weight_decay': 0}
        ]) 
    
    def _build_network(self, in_dim, out_dim, hidden, depth, act):
        return Network(input_size=in_dim, hidden_size=hidden, output_size=out_dim, depth=depth, act=act)
    
    # Define Loss Function
    def loss_diffusion_a(self):
        self.optimizer_diffusion_a.zero_grad()
        # PDE in domain
        c_a = self.model_diffusion_a(self.X_Int_A)[:,0]
        dc_a = torch.autograd.grad(c_a, self.X_Int_A, torch.ones_like(c_a), 
                                   retain_graph=True, create_graph=True)[0]
        dc_dx_a, dc_dt_a = dc_a[:,0], dc_a[:,1]
        dc_dxx_a = torch.autograd.grad(dc_dx_a, self.X_Int_A, torch.ones_like(dc_dx_a),
                                       retain_graph=True, create_graph=True)[0][:,0]

        dis_a = self.model_physics_a(self.X_Int_A)[:,0]
        dis_dx_a = torch.autograd.grad(dis_a, self.X_Int_A, torch.ones_like(dis_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1: # pure diffusion
            loss_diffusion_pde_a = self.criterion(dc_dt_a, D1*(dc_dxx_a + 1./self.X_Int_A[:,0]*dc_dx_a))
        else:
            # Calculate stress ->sigma ->dsigman-dx ->flux
            eps_xx_a = dis_dx_a
            eps_yy_a = dis_a / self.X_Int_A[:, 0]
            eps_zz_a = 0.
            # constitutive equations
            lam_a, theta_a = calc_lam_theta(E1, v1)
            stress_xx_a, stress_yy_a, stress_zz_a, sigman_a = calc_stress(
                eps_xx_a, eps_yy_a, eps_zz_a, c_a, lam_a, theta_a, omiga1, self.cycle_number
            )
            dsigman_dx_a = torch.autograd.grad(sigman_a, self.X_Int_A, 
                                               torch.ones_like(sigman_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            flux_a = calc_flux(dc_dx_a, c_a, dsigman_dx_a, D1, omiga1, Rg, T, self.cycle_number)
            flux_dx_a = torch.autograd.grad(flux_a, self.X_Int_A,
                                            torch.ones_like(flux_a),
                                            retain_graph=True, create_graph=True)[0][:,0]
            # Final diffusion equation (in cylindrical coordinate system)
            loss_diffusion_pde_a = self.criterion(dc_dt_a, -flux_dx_a - flux_a/self.X_Int_A[:,0])
            
        # ---2) boundary condition (left)---
        c_bnd_left = self.model_diffusion_a(self.XBnd_Left)[:,0]
        dc_bnd_left = torch.autograd.grad(c_bnd_left, self.XBnd_Left, 
                                          torch.ones_like(c_bnd_left),
                                          retain_graph=True, create_graph=True)[0][:,0]
        dis_bnd_left = self.model_physics_a(self.XBnd_Left)[:,0]
        dis_dx_bnd_left = torch.autograd.grad(dis_bnd_left, self.XBnd_Left,
                                              torch.ones_like(dis_bnd_left),
                                              retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            # D1 * dc_dx = 0 or J0
            loss_boundary_left = self.criterion(D1*dc_bnd_left, 
                                                self.XBnd_Left_Val*J0)
        else:
            # Calculate flux=0
            eps_bl_xx = dis_dx_bnd_left
            eps_bl_yy = dis_bnd_left/self.XBnd_Left[:,0]
            eps_bl_zz = 0.
            lam_a, theta_a = calc_lam_theta(E1, v1)
            # stress
            s_xx, s_yy, s_zz, sigman = calc_stress(eps_bl_xx, eps_bl_yy, eps_bl_zz,
                                                   c_bnd_left, lam_a, theta_a,
                                                   omiga1, self.cycle_number)
            dsigman_dx = torch.autograd.grad(sigman, self.XBnd_Left,
                                             torch.ones_like(sigman),
                                             retain_graph=True, create_graph=True)[0][:,0]
            flux_left = calc_flux(dc_bnd_left, c_bnd_left, dsigman_dx,
                                  D1, omiga1, Rg, T, self.cycle_number)
            flux_target = torch.zeros_like(flux_left)*J0
            loss_boundary_left = self.criterion(flux_left, flux_target)
        
        # ---3) initial condition---
        c_init_a = self.model_diffusion_a(self.XInit_A)[:,0]
        loss_initial_a = self.criterion(c_init_a, self.XInit_A_Val)
        
        # ---4) interface continuity---
        # Side A
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        dc_inf_a = torch.autograd.grad(c_inf_a, self.XInf_A,
                                       torch.ones_like(c_inf_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
        dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
        # Side B
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dc_inf_b = torch.autograd.grad(c_inf_b, self.XInf_B,
                                       torch.ones_like(c_inf_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        
        # flux
        if self.cycle_number == 1:
            diffusion_inf_a = -D1*dc_inf_a
            diffusion_inf_b = -D2*dc_inf_b
        else:
            # Side A
            eps_inf_xx_a = torch.autograd.grad(dis_inf_a, self.XInf_A, 
                                               torch.ones_like(dis_inf_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            lam_a, theta_a = calc_lam_theta(E1, v1)
            eps_yy_a = dis_inf_a/self.XInf_A[:,0]
            _, _, _, sigman_a = calc_stress(eps_inf_xx_a, eps_yy_a, 0., 
                                            c_inf_a, lam_a, theta_a, omiga1, self.cycle_number)
            dsigman_inf_a_dx = torch.autograd.grad(sigman_a, self.XInf_A,
                                                   torch.ones_like(sigman_a),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            diffusion_inf_a = calc_flux(dc_inf_a, c_inf_a, dsigman_inf_a_dx, 
                                        D1, omiga1, Rg, T, self.cycle_number)
            
            # Side B
            eps_inf_xx_b = torch.autograd.grad(dis_inf_b, self.XInf_B, 
                                               torch.ones_like(dis_inf_b),
                                               retain_graph=True, create_graph=True)[0][:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            eps_yy_b = dis_inf_b/self.XInf_B[:,0]
            _, _, _, sigman_b = calc_stress(eps_inf_xx_b, eps_yy_b, 0.,
                                            c_inf_b, lam_b, theta_b, omiga2, self.cycle_number)
            dsigman_inf_b_dx = torch.autograd.grad(sigman_b, self.XInf_B,
                                                   torch.ones_like(sigman_b),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            diffusion_inf_b = calc_flux(dc_inf_b, c_inf_b, dsigman_inf_b_dx, 
                                        D2, omiga2, Rg, T, self.cycle_number)

        loss_interface_c = self.criterion(c_inf_a, c_inf_b)
        loss_interface_j = self.criterion(diffusion_inf_a, diffusion_inf_b)
        
        # calculate the total loss
        loss_diffusion_a = self.awl_diffusion_a(
            loss_diffusion_pde_a, 
            loss_boundary_left, 
            loss_initial_a, 
            loss_interface_c, 
            loss_interface_j
        )
        loss_diffusion_a.backward()
        
        if self.iter_diffusion_a % 1000 == 0:
            print(f"[diffusion_a] iter={self.iter_diffusion_a}, Loss={loss_diffusion_a.item()}")
        self.iter_diffusion_a += 1
        
        return loss_diffusion_a
    
    def loss_diffusion_b(self):
        self.optimizer_diffusion_b.zero_grad()
        
        # PDE in domain
        c_b = self.model_diffusion_b(self.X_Int_B)[:,0]
        dc_b = torch.autograd.grad(c_b, self.X_Int_B, torch.ones_like(c_b),
                                   retain_graph=True, create_graph=True)[0]
        dc_dx_b, dc_dt_b = dc_b[:,0], dc_b[:,1]
        dc_dxx_b = torch.autograd.grad(dc_dx_b, self.X_Int_B, torch.ones_like(dc_dx_b),
                                       retain_graph=True, create_graph=True)[0][:,0]

        dis_b = self.model_physics_b(self.X_Int_B)[:,0]
        dis_dx_b = torch.autograd.grad(dis_b, self.X_Int_B, torch.ones_like(dis_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        
        if self.cycle_number == 1:
            loss_diffusion_pde_b = self.criterion(dc_dt_b, D2*(dc_dxx_b+1./self.X_Int_B[:,0]*dc_dx_b))
        else:
            eps_xx_b = dis_dx_b
            eps_yy_b = dis_b/self.X_Int_B[:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            stress_xx_b, stress_yy_b, stress_zz_b, sigman_b = calc_stress(
                eps_xx_b, eps_yy_b, 0., c_b, lam_b, theta_b, omiga2, self.cycle_number
            )
            dsigman_dx_b = torch.autograd.grad(sigman_b, self.X_Int_B,
                                               torch.ones_like(sigman_b),
                                               retain_graph=True, create_graph=True)[0][:,0]
            flux_b = calc_flux(dc_dx_b, c_b, dsigman_dx_b, D2, omiga2, Rg, T, self.cycle_number)
            flux_dx_b = torch.autograd.grad(flux_b, self.X_Int_B, 
                                            torch.ones_like(flux_b),
                                            retain_graph=True, create_graph=True)[0][:,0]
            loss_diffusion_pde_b = self.criterion(dc_dt_b, -flux_dx_b - flux_b/self.X_Int_B[:,0])
        
        # boundary
        c_bnd_right = self.model_diffusion_b(self.XBnd_Right)[:,0]
        dc_bnd_right = torch.autograd.grad(c_bnd_right, self.XBnd_Right, 
                                           torch.ones_like(c_bnd_right),
                                           retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            loss_boundary_right = self.criterion(D2*dc_bnd_right, self.XBnd_Right_Val*J0)
        else:
            dis_bnd_right = self.model_physics_b(self.XBnd_Right)[:,0]
            dis_dx_bnd_right = torch.autograd.grad(dis_bnd_right, self.XBnd_Right,
                                                   torch.ones_like(dis_bnd_right),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            eps_br_xx = dis_dx_bnd_right
            eps_br_yy = dis_bnd_right/self.XBnd_Right[:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            stress_bnd_right_xx, stress_bnd_right_yy, stress_bnd_right_zz, sigman_br = calc_stress(
                eps_br_xx, eps_br_yy, 0., c_bnd_right, lam_b, theta_b, omiga2, self.cycle_number
            )
            dsigman_bnd_right_dx = torch.autograd.grad(sigman_br, self.XBnd_Right,
                                                       torch.ones_like(sigman_br),
                                                       retain_graph=True, create_graph=True)[0][:,0]
            flux_bnd_right = calc_flux(dc_bnd_right, c_bnd_right, dsigman_bnd_right_dx,
                                       -D2, omiga2, Rg, T, self.cycle_number)
            flux_target = torch.ones_like(flux_bnd_right)*J0
            loss_boundary_right = self.criterion(flux_bnd_right, flux_target)
        
        # initial
        c_init_b = self.model_diffusion_b(self.XInit_B)[:,0]
        loss_initial_b = self.criterion(c_init_b, self.XInit_B_Val)
        
        # interface continuity
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dc_inf_a = torch.autograd.grad(c_inf_a, self.XInf_A, 
                                       torch.ones_like(c_inf_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        dc_inf_b = torch.autograd.grad(c_inf_b, self.XInf_B, 
                                       torch.ones_like(c_inf_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            diffusion_inf_a = -D1*dc_inf_a
            diffusion_inf_b = -D2*dc_inf_b
        else:
            dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
            dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
            eps_inf_xx_a = torch.autograd.grad(dis_inf_a, self.XInf_A, 
                                               torch.ones_like(dis_inf_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            lam_a, theta_a = calc_lam_theta(E1, v1)
            _, _, _, sigman_inf_a = calc_stress(eps_inf_xx_a, dis_inf_a/self.XInf_A[:,0], 0.,
                                                c_inf_a, lam_a, theta_a, omiga1, self.cycle_number)
            dsigman_inf_a_dx = torch.autograd.grad(sigman_inf_a, self.XInf_A,
                                                   torch.ones_like(sigman_inf_a),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            diffusion_inf_a = calc_flux(dc_inf_a, c_inf_a, dsigman_inf_a_dx, 
                                        D1, omiga1, Rg, T, self.cycle_number)

            eps_inf_xx_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                               torch.ones_like(dis_inf_b),
                                               retain_graph=True, create_graph=True)[0][:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            _, _, _, sigman_inf_b = calc_stress(eps_inf_xx_b, dis_inf_b/self.XInf_B[:,0], 0.,
                                                c_inf_b, lam_b, theta_b, omiga2, self.cycle_number)
            dsigman_inf_b_dx = torch.autograd.grad(sigman_inf_b, self.XInf_B,
                                                   torch.ones_like(sigman_inf_b),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            diffusion_inf_b = calc_flux(dc_inf_b, c_inf_b, dsigman_inf_b_dx, 
                                        D2, omiga2, Rg, T, self.cycle_number)

        loss_interface_c = self.criterion(c_inf_b, c_inf_a)
        loss_interface_j = self.criterion(diffusion_inf_b, diffusion_inf_a)

        # 合并
        loss_diffusion_b = self.awl_diffusion_b(
            loss_diffusion_pde_b,
            loss_boundary_right,
            loss_initial_b,
            loss_interface_c,
            loss_interface_j
        )
        loss_diffusion_b.backward()
        
        if self.iter_diffusion_b % 1000 == 0:
            print(f"[diffusion_b] iter={self.iter_diffusion_b}, Loss={loss_diffusion_b.item()}")
        self.iter_diffusion_b += 1

        return loss_diffusion_b
    
    def loss_physics_a(self):
        self.optimizer_physics_a.zero_grad()

        # PDE in domain
        c_a = self.model_diffusion_a(self.X_Int_A)[:,0]
        dis_a = self.model_physics_a(self.X_Int_A)[:,0]
        dis_dx_a = torch.autograd.grad(dis_a, self.X_Int_A, 
                                       torch.ones_like(dis_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        eps_xx_a = dis_dx_a
        eps_yy_a = dis_a/self.X_Int_A[:,0]
        eps_zz_a = 0.
        lam_a, theta_a = calc_lam_theta(E1, v1)
        c_eff = 2.0*c_a
        stress_xx_a, stress_yy_a, stress_zz_a, _ = calc_stress(
            eps_xx_a, eps_yy_a, eps_zz_a, c_eff, lam_a, theta_a, omiga1, 2 # cycle>1
        )
        
        # caculate energy density
        energy_density_a = 0.5*(eps_xx_a*stress_xx_a + eps_yy_a*stress_yy_a + eps_zz_a*stress_zz_a)
        weighted_energy_density_a = energy_density_a.flatten()*self.W_Int_A
        stress_energy_a = torch.sum(weighted_energy_density_a)
        
        # boundary condition (left)
        c_bnd_left = self.model_diffusion_a(self.XBnd_Left)[:,0]
        dis_bnd_left = self.model_physics_a(self.XBnd_Left)[:,0]
        dis_dx_bnd_left = torch.autograd.grad(dis_bnd_left, self.XBnd_Left,
                                              torch.ones_like(dis_bnd_left),
                                              retain_graph=True, create_graph=True)[0][:,0]
        eps_bl_xx = dis_dx_bnd_left
        eps_bl_yy = dis_bnd_left/self.XBnd_Left[:,0]
        # calc_stress_xx
        s_xx_bnd_left, _, _, _ = calc_stress(eps_bl_xx, eps_bl_yy, 0., c_bnd_left,
                                             lam_a, theta_a, omiga1, 2)
        zero_stress_left = torch.zeros_like(s_xx_bnd_left)
        loss_boundary_physics_left = self.criterion(s_xx_bnd_left, zero_stress_left)

        # displacement initial
        dis_init_a = self.model_physics_a(self.XInit_A)[:,0]
        loss_initial_physics_a = self.criterion(dis_init_a, self.XInit_A_Val)
        
        # interface
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
        dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]

        # Side A stress
        dis_dx_inf_a = torch.autograd.grad(dis_inf_a, self.XInf_A,
                                           torch.ones_like(dis_inf_a),
                                           retain_graph=True, create_graph=True)[0][:,0]
        eps_ia_xx = dis_dx_inf_a
        eps_ia_yy = dis_inf_a/self.XInf_A[:,0]
        s_ia_xx, _, _, _ = calc_stress(eps_ia_xx, eps_ia_yy, 0., c_inf_a,
                                       lam_a, theta_a, omiga1, 2)

        # Side B stress
        lam_b, theta_b = calc_lam_theta(E2, v2)
        dis_dx_inf_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                           torch.ones_like(dis_inf_b),
                                           retain_graph=True, create_graph=True)[0][:,0]
        eps_ib_xx = dis_dx_inf_b
        eps_ib_yy = dis_inf_b/self.XInf_B[:,0]
        s_ib_xx, _, _, _ = calc_stress(eps_ib_xx, eps_ib_yy, 0., c_inf_b,
                                       lam_b, theta_b, omiga2, 2)
        
        # Displacement continuity&stress continuity
        loss_interface_dis_a    = 1e2*self.criterion(dis_inf_a, dis_inf_b)
        loss_interface_stress_a = self.criterion(s_ia_xx, s_ib_xx)
        
        # caculation total loss
        loss_physics_a = self.awl_physics_a(
            stress_energy_a,
            loss_boundary_physics_left,
            loss_initial_physics_a,
            loss_interface_dis_a,
            loss_interface_stress_a
        )
        loss_physics_a.backward()
        
        if self.iter_physics_a % 1000 == 0:
            print(f"[physics_a] iter={self.iter_physics_a}, Loss={loss_physics_a.item()}")
        self.iter_physics_a += 1
        return loss_physics_a
    
    def loss_physics_b(self):
        self.optimizer_physics_b.zero_grad()
        
        c_b = self.model_diffusion_b(self.X_Int_B)[:,0]
        dis_b = self.model_physics_b(self.X_Int_B)[:,0]
        dis_dx_b = torch.autograd.grad(dis_b, self.X_Int_B,
                                       torch.ones_like(dis_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        eps_xx_b = dis_dx_b
        eps_yy_b = dis_b/self.X_Int_B[:,0]
        eps_zz_b = 0.
        lam_b, theta_b = calc_lam_theta(E2, v2)
        c_eff_b = 2.0*c_b
        stress_xx_b, stress_yy_b, stress_zz_b, _ = calc_stress(
            eps_xx_b, eps_yy_b, 0., c_eff_b, lam_b, theta_b, omiga2, 2
        )
        energy_density_b = 0.5*(eps_xx_b*stress_xx_b + eps_yy_b*stress_yy_b + eps_zz_b*stress_zz_b)
        weighted_energy_density_b = energy_density_b.flatten()*self.W_Int_B
        stress_energy_b = torch.sum(weighted_energy_density_b)

        # boundary
        c_bnd_right = self.model_diffusion_b(self.XBnd_Right)[:,0]
        dis_bnd_right = self.model_physics_b(self.XBnd_Right)[:,0]
        dis_dx_bnd_right = torch.autograd.grad(dis_bnd_right, self.XBnd_Right,
                                               torch.ones_like(dis_bnd_right),
                                               retain_graph=True, create_graph=True)[0][:,0]
        eps_br_xx = dis_dx_bnd_right
        eps_br_yy = dis_bnd_right/self.XBnd_Right[:,0]
        s_bnd_rx, _, _, _ = calc_stress(eps_br_xx, eps_br_yy, 0.,
                                        c_bnd_right, lam_b, theta_b, omiga2, 2)
        zero_stress_right = torch.zeros_like(s_bnd_rx)
        loss_boundary_physics_right = self.criterion(s_bnd_rx, zero_stress_right)

        # init
        dis_init_b = self.model_physics_b(self.XInit_B)[:,0]
        loss_initial_physics_b = self.criterion(dis_init_b, self.XInit_B_Val)

        # interface
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
        dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
        dis_dx_inf_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                           torch.ones_like(dis_inf_b),
                                           retain_graph=True, create_graph=True)[0][:,0]
        eps_ib_xx = dis_dx_inf_b
        eps_ib_yy = dis_inf_b/self.XInf_B[:,0]
        # Side A stress
        lam_a, theta_a = calc_lam_theta(E1, v1)
        dis_dx_inf_a = torch.autograd.grad(dis_inf_a, self.XInf_A,
                                           torch.ones_like(dis_inf_a),
                                           retain_graph=True, create_graph=True)[0][:,0]
        eps_ia_xx = dis_dx_inf_a
        eps_ia_yy = dis_inf_a/self.XInf_A[:,0]
        s_ia_xx, _, _, _ = calc_stress(eps_ia_xx, eps_ia_yy, 0., c_inf_a,
                                       lam_a, theta_a, omiga1, 2)
        # Side B stress
        s_ib_xx, _, _, _ = calc_stress(eps_ib_xx, eps_ib_yy, 0., c_inf_b,
                                       lam_b, theta_b, omiga2, 2)

        loss_interface_dis_b    = 1e2*self.criterion(dis_inf_b, dis_inf_a)
        loss_interface_stress_b = self.criterion(s_ib_xx, s_ia_xx)

        loss_physics_b = self.awl_physics_b(
            stress_energy_b,
            loss_boundary_physics_right,
            loss_initial_physics_b,
            loss_interface_dis_b,
            loss_interface_stress_b
        )
        loss_physics_b.backward()
        
        if self.iter_physics_b % 1000 == 0:
            print(f"[physics_b] iter={self.iter_physics_b}, Loss={loss_physics_b.item()}")
        self.iter_physics_b += 1

        return loss_physics_b
    
    # ------------------ Training ------------------
    def train(self, n_cycles = 10, steps=5001):
        self.model_diffusion_a.train()
        self.model_diffusion_b.train()
        self.model_physics_a.train()
        self.model_physics_b.train()
        
        for j in range(n_cycles):
            for step in range(steps):
                self.optimizer_diffusion_a.step(self.loss_diffusion_a)
                self.optimizer_diffusion_b.step(self.loss_diffusion_b)
                self.optimizer_physics_a.step(self.loss_physics_a)
                self.optimizer_physics_b.step(self.loss_physics_b)
            
            self.cycle_number += 1

if __name__ == "__main__":
    # Create model storage path
    save_dir = "./model/moving_interface/XDEM"
    os.makedirs(save_dir, exist_ok=True)
    # =============================================================================
    # Some fixed hyperparameters/physical constants/control quantities
    # =============================================================================
    config = load_config('./config.yaml')
    material_type = 'moving_interface'
    material = config['materials'][material_type]
    J0 = material['J0']
    Rg = material['Rg']
    T = material['T']
    material_a = material['material_a']
    material_b = material['material_b']
    D1 = material_a['D']
    v1 = material_a['v']
    omiga1 = material_a['omiga']
    E1 = material_a['E']
    D2 = material_b['D']
    v2 = material_b['v']
    omiga2 = material_b['omiga']
    E2 = material_b['E']
    # Set the number of Gauss-Legendre nodes
    n_gauss = 4
    num_elem_x, num_elem_t = 4, 4 
    num_elem_xA, num_elem_xB = 4, 4
    Diffusion_PINN = Multi_Diffusion()
    # =============================================================================
    # Set the number of iterations for parameter transmission
    # =============================================================================
    Diffusion_PINN.train(n_cycles = 10, steps=5001)
    # save model
    torch.save(Diffusion_PINN.model_diffusion_a, os.path.join(save_dir, "model_diffusion_a.pth"))
    torch.save(Diffusion_PINN.model_diffusion_b, os.path.join(save_dir, "model_diffusion_b.pth"))
    torch.save(Diffusion_PINN.model_physics_a, os.path.join(save_dir, "model_physics_a.pth"))
    torch.save(Diffusion_PINN.model_physics_b, os.path.join(save_dir, "model_physics_b.pth"))
    print("Models saved!")