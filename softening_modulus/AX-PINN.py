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

def build_model(in_dim, out_dim, hidden, depth, act_func):
    """
    Simple encapsulation of building networks for easy re-calls
    """
    return Network(input_size=in_dim, hidden_size=hidden, output_size=out_dim,
                   depth=depth, act=act_func)

def calc_E1_and_derivs(c: torch.Tensor, alpha: float):
    """
    E1(c) = 8 + 162*(1+αc)^(-1)
    Returns (E1, dE1/dc, d²E1/dc²), all tensors have the same shape as c
    """
    E1      = 8.0 + 162.0 * (1 + alpha*c).pow(-1)
    dE1_dc  = -162.0 * alpha * (1 + alpha*c).pow(-2)
    d2E1_dc =  324.0 * alpha**2 * (1 + alpha*c).pow(-3)
    return E1, dE1_dc, d2E1_dc

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

def calc_flux(dc_dx: torch.Tensor,
              c: torch.Tensor,
              dsig_dx: torch.Tensor,
              D: float,
              omiga: float,
              Rg: float,
              T: float,
              cycle_number: int,
              *,
              E: torch.Tensor | None = None,
              dE_dc: torch.Tensor | None = None,
              d2E_dc: torch.Tensor | None = None,
              sigman: torch.Tensor | None = None):
    
    if cycle_number == 1:
        return -D * dc_dx

    flux = -D * (dc_dx - omiga * c / (Rg * T) * dsig_dx)

    if E is not None and dE_dc is not None and d2E_dc is not None and sigman is not None:
        flux -= D * c / (Rg * T) * (
            - dE_dc.pow(2) / E.pow(3) * dc_dx * sigman.pow(2)
            + d2E_dc / (2 * E.pow(2)) * dc_dx * sigman.pow(2)
            + dE_dc  / (E.pow(2)) * sigman * dsig_dx
        )
    return flux

class Multi_Diffusion:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cycle_number = 1
        
        # Define Diffusion Neural Network A
        self.model_diffusion_a = build_model(2, 1, 32, 3, torch.nn.Tanh).to(self.device)
        self.model_diffusion_b = build_model(2, 1, 32, 3, torch.nn.Tanh).to(self.device)
        self.model_physics_a   = build_model(2, 1, 32, 3, lambda: torch.nn.SiLU()).to(self.device)
        self.model_physics_b   = build_model(2, 1, 32, 3, lambda: torch.nn.SiLU()).to(self.device)
        
        # creat geometric
        self.h = 0.03
        self.k = 0.06

        x_a = torch.arange(0.5, interface + self.h, self.h)
        t_a = torch.arange(0.0, 1.0 + self.k, self.k)
        x_b = torch.arange(interface, 1.0 + self.h, self.h)
        t_b = torch.arange(0.0, 1.0 + self.k, self.k)

        # interior
        self.X_Int_A = torch.stack(torch.meshgrid(x_a, t_a, indexing='ij'), dim=-1).reshape(-1, 2)
        self.X_Int_B = torch.stack(torch.meshgrid(x_b, t_b, indexing='ij'), dim=-1).reshape(-1, 2)
        # boundary
        self.XBnd_Left  = torch.stack(torch.meshgrid(torch.tensor([x_a[0]]), t_a, indexing='ij'), dim=-1).reshape(-1, 2)
        self.XBnd_Right = torch.stack(torch.meshgrid(torch.tensor([x_b[-1]]), t_b, indexing='ij'), dim=-1).reshape(-1, 2)
        # init
        self.XInit_A = torch.stack(torch.meshgrid(x_a, torch.tensor([t_a[0]]), indexing='ij'), dim=-1).reshape(-1, 2)
        self.XInit_B = torch.stack(torch.meshgrid(x_b, torch.tensor([t_b[0]]), indexing='ij'), dim=-1).reshape(-1, 2)
        # interface
        self.XInf_A = torch.stack(torch.meshgrid(torch.tensor([x_a[-1]]), t_a, indexing='ij'), dim=-1).reshape(-1, 2)
        self.XInf_B = torch.stack(torch.meshgrid(torch.tensor([x_b[0]]), t_b, indexing='ij'), dim=-1).reshape(-1, 2)

        # boundary & initial value
        self.XBnd_Left_Val  = torch.zeros(len(self.XBnd_Left), 1)
        self.XBnd_Right_Val = torch.ones(len(self.XBnd_Right), 1)
        self.XInit_A_Val    = torch.zeros(len(self.XInit_A),   1)
        self.XInit_B_Val    = torch.zeros(len(self.XInit_B),   1)

        all_pts = [self.X_Int_A, self.X_Int_B,
                   self.XBnd_Left, self.XBnd_Right,
                   self.XInit_A, self.XInit_B,
                   self.XInf_A, self.XInf_B]
        for p in all_pts:
            p.requires_grad_(True)
        tensor_list = all_pts + [
            self.XBnd_Left_Val, self.XBnd_Right_Val, 
            self.XInit_A_Val,   self.XInit_B_Val
        ]
        for t_ in tensor_list:
            t_ = t_.to(self.device)
            
        self.X_Int_A       = self.X_Int_A.to(self.device)
        self.X_Int_B       = self.X_Int_B.to(self.device)
        self.XBnd_Left     = self.XBnd_Left.to(self.device)
        self.XBnd_Right    = self.XBnd_Right.to(self.device)
        self.XInit_A       = self.XInit_A.to(self.device)
        self.XInit_B       = self.XInit_B.to(self.device)
        self.XInf_A        = self.XInf_A.to(self.device)
        self.XInf_B        = self.XInf_B.to(self.device)
        self.XBnd_Left_Val = self.XBnd_Left_Val.to(self.device)
        self.XBnd_Right_Val= self.XBnd_Right_Val.to(self.device)
        self.XInit_A_Val   = self.XInit_A_Val.to(self.device)
        self.XInit_B_Val   = self.XInit_B_Val.to(self.device)
        
        self.criterion = torch.nn.MSELoss()

        self.iter_diffusion_a = 0
        self.iter_diffusion_b = 0
        self.iter_physics_a = 0
        self.iter_physics_b = 0
        
        # Set Loss Weighting Function
        self.awl_diffusion_a = AutomaticWeightedLoss(5)
        self.awl_diffusion_b = AutomaticWeightedLoss(5)
        self.awl_physics_a = AutomaticWeightedLoss(5)
        self.awl_physics_b = AutomaticWeightedLoss(5)
        
        # Set Adam Optimizer
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
            E1, dE1_dc, d2E1_dc = calc_E1_and_derivs(c_a, alpha)
            lam_a, theta_a = calc_lam_theta(E1, v1)
            stress_xx_a, stress_yy_a, stress_zz_a, sigman_a = calc_stress(
                eps_xx_a, eps_yy_a, eps_zz_a, c_a, lam_a, theta_a, omiga1, self.cycle_number
            )
            dsigman_dx_a = torch.autograd.grad(sigman_a, self.X_Int_A, 
                                               torch.ones_like(sigman_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            flux_a = calc_flux(dc_dx=dc_dx_a,
                   c=c_a,
                   dsig_dx=dsigman_dx_a,
                   D=D1,
                   omiga=omiga1,
                   Rg=Rg,
                   T=T,
                   cycle_number=self.cycle_number,
                   E=E1,
                   dE_dc=dE1_dc,
                   d2E_dc=d2E1_dc,
                   sigman=sigman_a)
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
            E1_bnd, dE1_dc_bnd, d2E1_dc_bnd = calc_E1_and_derivs(c_bnd_left, alpha)
            lam_a, theta_a = calc_lam_theta(E1_bnd, v1)
            # stress
            s_xx, s_yy, s_zz, sigman = calc_stress(eps_bl_xx, eps_bl_yy, eps_bl_zz,
                                                   c_bnd_left, lam_a, theta_a,
                                                   omiga1, self.cycle_number)
            dsigman_dx = torch.autograd.grad(sigman, self.XBnd_Left,
                                             torch.ones_like(sigman),
                                             retain_graph=True, create_graph=True)[0][:,0]
            flux_left = calc_flux(dc_dx=dc_bnd_left,
                   c=c_bnd_left,
                   dsig_dx=dsigman_dx,
                   D=D1,
                   omiga=omiga1,
                   Rg=Rg,
                   T=T,
                   cycle_number=self.cycle_number,
                   E=E1_bnd,
                   dE_dc=dE1_dc_bnd,
                   d2E_dc=d2E1_dc_bnd,
                   sigman=sigman)
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
            E1_inf, dE1_dc_inf, d2E1_dc_inf = calc_E1_and_derivs(c_inf_a, alpha)
            lam_a, theta_a = calc_lam_theta(E1_inf, v1)
            eps_yy_a = dis_inf_a/self.XInf_A[:,0]
            _, _, _, sigman_a = calc_stress(eps_inf_xx_a, eps_yy_a, 0., 
                                            c_inf_a, lam_a, theta_a, omiga1, self.cycle_number)
            dsigman_inf_a_dx = torch.autograd.grad(sigman_a, self.XInf_A,
                                                   torch.ones_like(sigman_a),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            diffusion_inf_a = calc_flux(dc_dx=dc_inf_a,
                   c=c_inf_a,
                   dsig_dx=dsigman_inf_a_dx,
                   D=D1,
                   omiga=omiga1,
                   Rg=Rg,
                   T=T,
                   cycle_number=self.cycle_number,
                   E=E1_inf,
                   dE_dc=dE1_dc_inf,
                   d2E_dc=d2E1_dc_inf,
                   sigman=sigman_a)
            
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
            E1_inf, dE1_dc_inf, d2E1_dc_inf = calc_E1_and_derivs(c_inf_a, alpha)
            lam_a, theta_a = calc_lam_theta(E1_inf, v1)
            _, _, _, sigman_inf_a = calc_stress(eps_inf_xx_a, dis_inf_a/self.XInf_A[:,0], 0.,
                                                c_inf_a, lam_a, theta_a, omiga1, self.cycle_number)
            dsigman_inf_a_dx = torch.autograd.grad(sigman_inf_a, self.XInf_A,
                                                   torch.ones_like(sigman_inf_a),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            diffusion_inf_a = calc_flux(dc_dx=dc_inf_a,
                   c=c_inf_a,
                   dsig_dx=dsigman_inf_a_dx,
                   D=D1,
                   omiga=omiga1,
                   Rg=Rg,
                   T=T,
                   cycle_number=self.cycle_number,
                   E=E1_inf,
                   dE_dc=dE1_dc_inf,
                   d2E_dc=d2E1_dc_inf,
                   sigman=sigman_inf_a)

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
        E1, _, _ = calc_E1_and_derivs(c_a, alpha)
        lam_a, theta_a = calc_lam_theta(E1, v1)
        c_eff = 2.0*c_a
        stress_xx_a, stress_yy_a, stress_zz_a, _ = calc_stress(
            eps_xx_a, eps_yy_a, eps_zz_a, c_eff, lam_a, theta_a, omiga1, 2 # cycle>1
        )
        
        # 平衡方程
        dsxx_dx_a = torch.autograd.grad(stress_xx_a, self.X_Int_A,
                                        torch.ones_like(stress_xx_a),
                                        retain_graph=True, create_graph=True)[0][:,0]
    
        loss_interior_a = self.criterion(dsxx_dx_a, -1.0*(stress_xx_a - stress_yy_a)/self.X_Int_A[:,0])
        
        # boundary condition (left)
        c_bnd_left = self.model_diffusion_a(self.XBnd_Left)[:,0]
        dis_bnd_left = self.model_physics_a(self.XBnd_Left)[:,0]
        dis_dx_bnd_left = torch.autograd.grad(dis_bnd_left, self.XBnd_Left,
                                              torch.ones_like(dis_bnd_left),
                                              retain_graph=True, create_graph=True)[0][:,0]
        eps_bl_xx = dis_dx_bnd_left
        eps_bl_yy = dis_bnd_left/self.XBnd_Left[:,0]
        # calc_stress_xx
        E1_bnd_left, _, _ = calc_E1_and_derivs(c_bnd_left, alpha)
        lam_bnd_left, theta_bnd_left = calc_lam_theta(E1_bnd_left, v1)
        s_xx_bnd_left, _, _, _ = calc_stress(eps_bl_xx, eps_bl_yy, 0., c_bnd_left,
                                     lam_bnd_left, theta_bnd_left, omiga1, 2)
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
        E1_inf, _, _ = calc_E1_and_derivs(c_inf_a, alpha)
        lam_a, theta_a = calc_lam_theta(E1_inf, v1)
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
            loss_interior_a,
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
        
        dsxx_dx_b = torch.autograd.grad(stress_xx_b, self.X_Int_B,
                                        torch.ones_like(stress_xx_b),
                                        retain_graph=True, create_graph=True)[0][:,0]
        loss_interior_b = self.criterion(dsxx_dx_b, -1.0*(stress_xx_b-stress_yy_b)/self.X_Int_B[:,0])

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
        E1_inf, _, _ = calc_E1_and_derivs(c_inf_a, alpha)
        lam_a, theta_a = calc_lam_theta(E1_inf, v1)
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
            loss_interior_b,
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
    
    def train(self, n_cycle=10, diffusion_steps=5001, physics_steps=5001):
        self.model_diffusion_a.train()
        self.model_diffusion_b.train()
        self.model_physics_a.train()
        self.model_physics_b.train()

        for j in range(n_cycle):
            print(f"\n===== Cycle_Number: {self.cycle_number} =====")

            # 1) Training Diffusion Field
            print(f"  [Cycle {self.cycle_number}] Training diffusion (Adam)...")
            for step in range(diffusion_steps):
                self.optimizer_diffusion_a.step(self.loss_diffusion_a)
                self.optimizer_diffusion_b.step(self.loss_diffusion_b)

            # 2) Training Displacement Field
            print(f"  [Cycle {self.cycle_number}] Training physics (Adam)...")
            for step in range(physics_steps):
                self.optimizer_physics_a.step(self.loss_physics_a)
                self.optimizer_physics_b.step(self.loss_physics_b)

            self.cycle_number += 1
            
if __name__ == "__main__":
    # Create model storage path
    save_dir = "./model/softening_modulus/AXPINN"
    os.makedirs(save_dir, exist_ok=True)
    # =============================================================================
    # Some fixed hyperparameters/physical constants/control quantities
    # =============================================================================
    config = load_config('./config.yaml')
    material_type = 'softening_modulus'
    material = config['materials'][material_type]
    interface = material['interface']
    J0 = material['J0']
    Rg = material['Rg']
    T = material['T']
    Cmax = material['Cmax']
    alpha = 4.4/Cmax
    material_a = material['material_a']
    material_b = material['material_b']
    D1 = material_a['D']
    v1 = material_a['v']
    omiga1 = material_a['omiga']
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
    Diffusion_PINN.train(n_cycle=10, diffusion_steps=5001, physics_steps=5001)
    # save model
    torch.save(Diffusion_PINN.model_diffusion_a, os.path.join(save_dir, "model_diffusion_a.pth"))
    torch.save(Diffusion_PINN.model_diffusion_b, os.path.join(save_dir, "model_diffusion_b.pth"))
    torch.save(Diffusion_PINN.model_physics_a, os.path.join(save_dir, "model_physics_a.pth"))
    torch.save(Diffusion_PINN.model_physics_b, os.path.join(save_dir, "model_physics_b.pth"))
    print("Models saved!")