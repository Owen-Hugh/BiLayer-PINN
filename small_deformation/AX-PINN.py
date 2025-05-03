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

def calc_lam_theta(E, v):
    """
    Given E, v, compute lam, theta (Lamé constant in the linear elasticity eigenstructure)
    """
    lam = v*E / ((1+v)*(1-2*v))
    theta = E/(1+v)
    return lam, theta

def compute_stress_polar(eps_xx, eps_yy, eps_zz, c, lam, theta, omiga, cycle=None):
    """
    Given the strains (eps_xx, eps_yy, eps_zz), and c, lam, theta, omiga.
    Returns stress_xx, stress_yy, stress_zz, and the average stress sigman.
    """
    # “- omiga * c” 部分(根据具体公式可以修改)
    stress_xx = lam*(eps_xx + eps_yy + eps_zz - omiga*c) + theta*(eps_xx - 1/3*omiga*c)
    stress_yy = lam*(eps_xx + eps_yy + eps_zz - omiga*c) + theta*(eps_yy - 1/3*omiga*c)
    stress_zz = lam*(eps_xx + eps_yy + eps_zz - omiga*c) + theta*(eps_zz - 1/3*omiga*c)
    sigman    = (stress_xx + stress_yy + stress_zz)/3
    return stress_xx, stress_yy, stress_zz, sigman

def ensure_dir(dir_path):
    """If the destination folder does not exist, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
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
        
        c_a = self.model_diffusion_a(self.X_Int_A)[:, 0]
        dc_a = torch.autograd.grad(c_a, self.X_Int_A, 
                                   torch.ones_like(c_a),
                                   retain_graph=True, create_graph=True)[0]
        dc_dx_a, dc_dt_a = dc_a[:,0], dc_a[:,1]
        dc_dxx_a = torch.autograd.grad(dc_dx_a, self.X_Int_A, 
                                       torch.ones_like(dc_dx_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        
        dis_a = self.model_physics_a(self.X_Int_A)[:,0]
        dis_dx_a = torch.autograd.grad(dis_a, self.X_Int_A,
                                       torch.ones_like(dis_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        
        if self.cycle_number == 1:
            loss_pde_a = self.criterion(dc_dt_a, D1*(dc_dxx_a + 1./self.X_Int_A[:,0]*dc_dx_a))
        else:
            # 耦合项
            eps_xx_a = dis_dx_a
            eps_yy_a = dis_a / self.X_Int_A[:,0]
            lam_a, theta_a = calc_lam_theta(E1, v1)
            # 应力
            s_xx, s_yy, s_zz, sigman_a = compute_stress_polar(
                eps_xx_a, eps_yy_a, 0., c_a, lam_a, theta_a, omiga1
            )
            dsigman_dx_a = torch.autograd.grad(sigman_a, self.X_Int_A,
                                               torch.ones_like(sigman_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            flux_a = -D1*(dc_dx_a - omiga1*c_a/(Rg*T)*dsigman_dx_a)
            flux_dx_a = torch.autograd.grad(flux_a, self.X_Int_A,
                                            torch.ones_like(flux_a),
                                            retain_graph=True, create_graph=True)[0][:,0]
            loss_pde_a = self.criterion(dc_dt_a, -flux_dx_a - flux_a/self.X_Int_A[:,0])

        # 2) boundary condition (left side)
        c_bnd_left = self.model_diffusion_a(self.XBnd_Left)[:,0]
        dc_bnd_left = torch.autograd.grad(c_bnd_left, self.XBnd_Left,
                                          torch.ones_like(c_bnd_left),
                                          retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            loss_bnd_left = self.criterion(D1*dc_bnd_left, self.XBnd_Left_Val*J0)
        else:
            dis_bnd_left = self.model_physics_a(self.XBnd_Left)[:,0]
            dis_dx_bnd_left = torch.autograd.grad(dis_bnd_left, self.XBnd_Left,
                                                  torch.ones_like(dis_bnd_left),
                                                  retain_graph=True, create_graph=True)[0][:,0]
            eps_xx_bl = dis_dx_bnd_left
            eps_yy_bl = dis_bnd_left/self.XBnd_Left[:,0]
            lam_a, theta_a = calc_lam_theta(E1, v1)
            s_xx_bl, s_yy_bl, s_zz_bl, sigman_bl = compute_stress_polar(
                eps_xx_bl, eps_yy_bl, 0., c_bnd_left, lam_a, theta_a, omiga1
            )
            dsigman_bl_dx = torch.autograd.grad(sigman_bl, self.XBnd_Left,
                                                torch.ones_like(sigman_bl),
                                                retain_graph=True, create_graph=True)[0][:,0]
            flux_left = D1*(dc_bnd_left - omiga1*c_bnd_left/(Rg*T)*dsigman_bl_dx)
            flux_target = torch.zeros_like(flux_left)*J0
            loss_bnd_left = self.criterion(flux_left, flux_target)

        # 3) initial condition
        c_init_a = self.model_diffusion_a(self.XInit_A)[:,0]
        loss_init_a = self.criterion(c_init_a, self.XInit_A_Val)

        # 4) interface
        #   A侧
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        dc_inf_a = torch.autograd.grad(c_inf_a, self.XInf_A,
                                       torch.ones_like(c_inf_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            flux_inf_a = -D1*dc_inf_a
        else:
            dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
            dis_dx_inf_a = torch.autograd.grad(dis_inf_a, self.XInf_A,
                                               torch.ones_like(dis_inf_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            eps_xx_inf_a = dis_dx_inf_a
            eps_yy_inf_a = dis_inf_a/self.XInf_A[:,0]
            lam_a, theta_a = calc_lam_theta(E1, v1)
            _, _, _, sigman_inf_a = compute_stress_polar(eps_xx_inf_a, eps_yy_inf_a, 0.,
                                                        c_inf_a, lam_a, theta_a, omiga1)
            dsigman_inf_a_dx = torch.autograd.grad(sigman_inf_a, self.XInf_A,
                                                   torch.ones_like(sigman_inf_a),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            flux_inf_a = -D1*(dc_inf_a - omiga1*c_inf_a/(Rg*T)*dsigman_inf_a_dx)

        #   B侧
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dc_inf_b = torch.autograd.grad(c_inf_b, self.XInf_B,
                                       torch.ones_like(c_inf_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            flux_inf_b = -D2*dc_inf_b
        else:
            dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
            dis_dx_inf_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                               torch.ones_like(dis_inf_b),
                                               retain_graph=True, create_graph=True)[0][:,0]
            eps_xx_inf_b = dis_dx_inf_b
            eps_yy_inf_b = dis_inf_b/self.XInf_B[:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            _, _, _, sigman_inf_b = compute_stress_polar(eps_xx_inf_b, eps_yy_inf_b, 0.,
                                                        c_inf_b, lam_b, theta_b, omiga2)
            dsigman_inf_b_dx = torch.autograd.grad(sigman_inf_b, self.XInf_B,
                                                   torch.ones_like(sigman_inf_b),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            flux_inf_b = -D2*(dc_inf_b - omiga2*c_inf_b/(Rg*T)*dsigman_inf_b_dx)
        
        loss_interface_c = self.criterion(c_inf_a, c_inf_b)
        loss_interface_j = self.criterion(flux_inf_a, flux_inf_b)
        
        # 合并
        loss_diffusion_a = self.awl_diffusion_a(
            loss_pde_a,         # PDE
            loss_bnd_left,      # BC
            loss_init_a,        # IC
            loss_interface_c,   # continuity of c
            loss_interface_j    # continuity of flux
        )
        loss_diffusion_a.backward()

        if self.iter_diffusion_a % 1000 == 0:
            print(f"[Diffusion_A] iter={self.iter_diffusion_a}, loss={loss_diffusion_a.item():.3e}")
        self.iter_diffusion_a += 1
        return loss_diffusion_a

    def loss_diffusion_b(self):
        self.optimizer_diffusion_b.zero_grad()
        
        # 1) PDE
        c_b = self.model_diffusion_b(self.X_Int_B)[:,0]
        dc_b = torch.autograd.grad(c_b, self.X_Int_B,
                                   torch.ones_like(c_b),
                                   retain_graph=True, create_graph=True)[0]
        dc_dx_b, dc_dt_b = dc_b[:,0], dc_b[:,1]
        dc_dxx_b = torch.autograd.grad(dc_dx_b, self.X_Int_B,
                                       torch.ones_like(dc_dx_b),
                                       retain_graph=True, create_graph=True)[0][:,0]

        dis_b = self.model_physics_b(self.X_Int_B)[:,0]
        dis_dx_b = torch.autograd.grad(dis_b, self.X_Int_B,
                                       torch.ones_like(dis_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            loss_pde_b = self.criterion(dc_dt_b, D2*(dc_dxx_b + 1./self.X_Int_B[:,0]*dc_dx_b))
        else:
            eps_xx_b = dis_dx_b
            eps_yy_b = dis_b/self.X_Int_B[:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            s_xx_b, s_yy_b, s_zz_b, sigman_b = compute_stress_polar(
                eps_xx_b, eps_yy_b, 0., c_b, lam_b, theta_b, omiga2
            )
            dsigman_dx_b = torch.autograd.grad(sigman_b, self.X_Int_B,
                                               torch.ones_like(sigman_b),
                                               retain_graph=True, create_graph=True)[0][:,0]
            flux_b = -D2*(dc_dx_b - omiga2*c_b/(Rg*T)*dsigman_dx_b)
            flux_dx_b = torch.autograd.grad(flux_b, self.X_Int_B,
                                            torch.ones_like(flux_b),
                                            retain_graph=True, create_graph=True)[0][:,0]
            loss_pde_b = self.criterion(dc_dt_b, -flux_dx_b - flux_b/self.X_Int_B[:,0])

        # 2) BC (right)
        c_bnd_right = self.model_diffusion_b(self.XBnd_Right)[:,0]
        dc_bnd_right = torch.autograd.grad(c_bnd_right, self.XBnd_Right,
                                           torch.ones_like(c_bnd_right),
                                           retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            loss_bnd_right = self.criterion(D2*dc_bnd_right, self.XBnd_Right_Val*J0)
        else:
            dis_bnd_right = self.model_physics_b(self.XBnd_Right)[:,0]
            dis_dx_bnd_right = torch.autograd.grad(dis_bnd_right, self.XBnd_Right,
                                                   torch.ones_like(dis_bnd_right),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            eps_xx_br = dis_dx_bnd_right
            eps_yy_br = dis_bnd_right/self.XBnd_Right[:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            s_xx_br, s_yy_br, s_zz_br, sigman_br = compute_stress_polar(
                eps_xx_br, eps_yy_br, 0., c_bnd_right, lam_b, theta_b, omiga2
            )
            dsigman_br_dx = torch.autograd.grad(sigman_br, self.XBnd_Right,
                                                torch.ones_like(sigman_br),
                                                retain_graph=True, create_graph=True)[0][:,0]
            flux_right = D2*(dc_bnd_right - omiga2*c_bnd_right/(Rg*T)*dsigman_br_dx)
            flux_target = torch.ones_like(flux_right)*J0
            loss_bnd_right = self.criterion(flux_right, flux_target)

        # 3) IC
        c_init_b = self.model_diffusion_b(self.XInit_B)[:,0]
        loss_init_b = self.criterion(c_init_b, self.XInit_B_Val)

        # 4) interface
        #   A侧
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        dc_inf_a = torch.autograd.grad(c_inf_a, self.XInf_A,
                                       torch.ones_like(c_inf_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            flux_inf_a = -D1*dc_inf_a
        else:
            dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
            dis_dx_inf_a = torch.autograd.grad(dis_inf_a, self.XInf_A,
                                               torch.ones_like(dis_inf_a),
                                               retain_graph=True, create_graph=True)[0][:,0]
            eps_xx_inf_a = dis_dx_inf_a
            eps_yy_inf_a = dis_inf_a/self.XInf_A[:,0]
            lam_a, theta_a = calc_lam_theta(E1, v1)
            _, _, _, sigman_inf_a = compute_stress_polar(eps_xx_inf_a, eps_yy_inf_a, 0.,
                                                        c_inf_a, lam_a, theta_a, omiga1)
            dsigman_inf_a_dx = torch.autograd.grad(sigman_inf_a, self.XInf_A,
                                                   torch.ones_like(sigman_inf_a),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            flux_inf_a = -D1*(dc_inf_a - omiga1*c_inf_a/(Rg*T)*dsigman_inf_a_dx)

        #   B侧
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dc_inf_b = torch.autograd.grad(c_inf_b, self.XInf_B,
                                       torch.ones_like(c_inf_b),
                                       retain_graph=True, create_graph=True)[0][:,0]
        if self.cycle_number == 1:
            flux_inf_b = -D2*dc_inf_b
        else:
            dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
            dis_dx_inf_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                               torch.ones_like(dis_inf_b),
                                               retain_graph=True, create_graph=True)[0][:,0]
            eps_xx_inf_b = dis_dx_inf_b
            eps_yy_inf_b = dis_inf_b/self.XInf_B[:,0]
            lam_b, theta_b = calc_lam_theta(E2, v2)
            _, _, _, sigman_inf_b = compute_stress_polar(eps_xx_inf_b, eps_yy_inf_b, 0.,
                                                        c_inf_b, lam_b, theta_b, omiga2)
            dsigman_inf_b_dx = torch.autograd.grad(sigman_inf_b, self.XInf_B,
                                                   torch.ones_like(sigman_inf_b),
                                                   retain_graph=True, create_graph=True)[0][:,0]
            flux_inf_b = -D2*(dc_inf_b - omiga2*c_inf_b/(Rg*T)*dsigman_inf_b_dx)
        
        loss_interface_c = self.criterion(c_inf_b, c_inf_a)
        loss_interface_j = self.criterion(flux_inf_b, flux_inf_a)

        loss_diffusion_b = self.awl_diffusion_b(
            loss_pde_b,
            loss_bnd_right,
            loss_init_b,
            loss_interface_c,
            loss_interface_j
        )
        loss_diffusion_b.backward()

        if self.iter_diffusion_b % 1000 == 0:
            print(f"[Diffusion_B] iter={self.iter_diffusion_b}, loss={loss_diffusion_b.item():.3e}")
        self.iter_diffusion_b += 1
        return loss_diffusion_b

    def loss_physics_a(self):
        self.optimizer_physics_a.zero_grad()

        # domain
        c_a = self.model_diffusion_a(self.X_Int_A)[:,0]
        dis_a = self.model_physics_a(self.X_Int_A)[:,0]
        dis_dx_a = torch.autograd.grad(dis_a, self.X_Int_A,
                                       torch.ones_like(dis_a),
                                       retain_graph=True, create_graph=True)[0][:,0]
        eps_xx_a = dis_dx_a
        eps_yy_a = dis_a/self.X_Int_A[:,0]
        
        lam_a, theta_a = calc_lam_theta(E1, v1)
        s_xx_a, s_yy_a, s_zz_a, _ = compute_stress_polar(
            eps_xx_a, eps_yy_a, 0., c_a, lam_a, theta_a, omiga1
        )
        # 平衡方程
        dsxx_dx_a = torch.autograd.grad(s_xx_a, self.X_Int_A,
                                        torch.ones_like(s_xx_a),
                                        retain_graph=True, create_graph=True)[0][:,0]
    
        loss_interior_a = self.criterion(dsxx_dx_a, -1.0*(s_xx_a - s_yy_a)/self.X_Int_A[:,0])

        # boundary
        c_bnd_left = self.model_diffusion_a(self.XBnd_Left)[:,0]
        dis_bnd_left = self.model_physics_a(self.XBnd_Left)[:,0]
        dis_dx_bnd_left = torch.autograd.grad(dis_bnd_left, self.XBnd_Left,
                                              torch.ones_like(dis_bnd_left),
                                              retain_graph=True, create_graph=True)[0][:,0]
        eps_bnd_xx_left = dis_dx_bnd_left
        eps_bnd_yy_left = dis_bnd_left/self.XBnd_Left[:,0]
        s_bnd_xx_left, _, _, _ = compute_stress_polar(
            eps_bnd_xx_left, eps_bnd_yy_left, 0., c_bnd_left, lam_a, theta_a, omiga1
        )
        # stress=0
        loss_bnd_left = self.criterion(s_bnd_xx_left, torch.zeros_like(s_bnd_xx_left))

        # initial
        dis_init_a = self.model_physics_a(self.XInit_A)[:,0]
        loss_init_a = self.criterion(dis_init_a, self.XInit_A_Val)

        # interface
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
        dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
        dis_dx_inf_a = torch.autograd.grad(dis_inf_a, self.XInf_A,
                                           torch.ones_like(dis_inf_a),
                                           retain_graph=True, create_graph=True)[0][:,0]
        # B侧
        lam_b, theta_b = calc_lam_theta(E2, v2)
        dis_dx_inf_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                           torch.ones_like(dis_inf_b),
                                           retain_graph=True, create_graph=True)[0][:,0]
        eps_inf_xx_a = dis_dx_inf_a
        eps_inf_yy_a = dis_inf_a/self.XInf_A[:,0]
        eps_inf_xx_b = dis_dx_inf_b
        eps_inf_yy_b = dis_inf_b/self.XInf_B[:,0]
        s_inf_xx_a, _, _, _ = compute_stress_polar(
            eps_inf_xx_a, eps_inf_yy_a, 0., c_inf_a, lam_a, theta_a, omiga1
        )
        s_inf_xx_b, _, _, _ = compute_stress_polar(
            eps_inf_xx_b, eps_inf_yy_b, 0., c_inf_b, lam_b, theta_b, omiga2
        )
        loss_interface_dis_a    = 1e2*self.criterion(dis_inf_a, dis_inf_b)
        loss_interface_stress_a = self.criterion(s_inf_xx_a, s_inf_xx_b)

        # AWL
        loss_physics_a = self.awl_physics_a(
            loss_interior_a,
            loss_bnd_left,
            loss_init_a,
            loss_interface_dis_a,
            loss_interface_stress_a
        )
        loss_physics_a.backward()

        if self.iter_physics_a % 1000 == 0:
            print(f"[Physics_A] iter={self.iter_physics_a}, loss={loss_physics_a.item():.3e}")
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
        lam_b, theta_b = calc_lam_theta(E2, v2)
        s_xx_b, s_yy_b, s_zz_b, _ = compute_stress_polar(
            eps_xx_b, eps_yy_b, 0., c_b, lam_b, theta_b, omiga2
        )
        dsxx_dx_b = torch.autograd.grad(s_xx_b, self.X_Int_B,
                                        torch.ones_like(s_xx_b),
                                        retain_graph=True, create_graph=True)[0][:,0]
        loss_interior_b = self.criterion(dsxx_dx_b, -1.0*(s_xx_b-s_yy_b)/self.X_Int_B[:,0])

        # boundary
        c_bnd_right = self.model_diffusion_b(self.XBnd_Right)[:,0]
        dis_bnd_right = self.model_physics_b(self.XBnd_Right)[:,0]
        dis_dx_bnd_right = torch.autograd.grad(dis_bnd_right, self.XBnd_Right,
                                               torch.ones_like(dis_bnd_right),
                                               retain_graph=True, create_graph=True)[0][:,0]
        eps_bnd_xx_right = dis_dx_bnd_right
        eps_bnd_yy_right = dis_bnd_right/self.XBnd_Right[:,0]
        s_bnd_xx_right, _, _, _ = compute_stress_polar(
            eps_bnd_xx_right, eps_bnd_yy_right, 0., c_bnd_right, lam_b, theta_b, omiga2
        )
        loss_bnd_right = self.criterion(s_bnd_xx_right, torch.zeros_like(s_bnd_xx_right))

        # initial
        dis_init_b = self.model_physics_b(self.XInit_B)[:,0]
        loss_init_b = self.criterion(dis_init_b, self.XInit_B_Val)

        # interface
        c_inf_a = self.model_diffusion_a(self.XInf_A)[:,0]
        c_inf_b = self.model_diffusion_b(self.XInf_B)[:,0]
        dis_inf_a = self.model_physics_a(self.XInf_A)[:,0]
        dis_inf_b = self.model_physics_b(self.XInf_B)[:,0]
        dis_dx_inf_a = torch.autograd.grad(dis_inf_a, self.XInf_A,
                                           torch.ones_like(dis_inf_a),
                                           retain_graph=True, create_graph=True)[0][:,0]
        dis_dx_inf_b = torch.autograd.grad(dis_inf_b, self.XInf_B,
                                           torch.ones_like(dis_inf_b),
                                           retain_graph=True, create_graph=True)[0][:,0]
        lam_a, theta_a = calc_lam_theta(E1, v1)
        eps_inf_xx_a = dis_dx_inf_a
        eps_inf_yy_a = dis_inf_a/self.XInf_A[:,0]
        eps_inf_xx_b = dis_dx_inf_b
        eps_inf_yy_b = dis_inf_b/self.XInf_B[:,0]
        s_inf_xx_a, _, _, _ = compute_stress_polar(
            eps_inf_xx_a, eps_inf_yy_a, 0., c_inf_a, lam_a, theta_a, omiga1
        )
        s_inf_xx_b, _, _, _ = compute_stress_polar(
            eps_inf_xx_b, eps_inf_yy_b, 0., c_inf_b, lam_b, theta_b, omiga2
        )
        loss_interface_dis_b    = 1e2*self.criterion(dis_inf_b, dis_inf_a)
        loss_interface_stress_b = self.criterion(s_inf_xx_b, s_inf_xx_a)

        loss_physics_b = self.awl_physics_b(
            loss_interior_b,
            loss_bnd_right,
            loss_init_b,
            loss_interface_dis_b,
            loss_interface_stress_b
        )
        loss_physics_b.backward()

        if self.iter_physics_b % 1000 == 0:
            print(f"[Physics_B] iter={self.iter_physics_b}, loss={loss_physics_b.item():.3e}")
        self.iter_physics_b += 1
        return loss_physics_b

    # ------------------ Training ------------------
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
    save_dir = "./model/small_deformation/AXPINN"
    # =============================================================================
    # Some fixed hyperparameters/physical constants/control quantities
    # =============================================================================
    config = load_config('./config.yaml')
    material_type = 'small_deformation'
    material = config['materials'][material_type]
    interface = material['interface']
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
    Diffusion_PINN = Multi_Diffusion()
    # =============================================================================
    # Set the number of iterations for parameter transmission
    # =============================================================================
    Diffusion_PINN.train(n_cycle=10, diffusion_steps=5001, physics_steps=5001)
    # save model
    ensure_dir(save_dir)
    torch.save(Diffusion_PINN.model_diffusion_a, os.path.join(save_dir, "model_diffusion_a.pth"))
    torch.save(Diffusion_PINN.model_diffusion_b, os.path.join(save_dir, "model_diffusion_b.pth"))
    torch.save(Diffusion_PINN.model_physics_a,   os.path.join(save_dir, "model_physics_a.pth"))
    torch.save(Diffusion_PINN.model_physics_b,   os.path.join(save_dir, "model_physics_b.pth"))
    print("Models saved!")