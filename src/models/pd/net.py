import torch as tc
import torch.nn as nn
from src.models.pd.algo import PD_Standalone_algo

class PD_Standalone_layer(nn.Module):
    def __init__(self, init_tau, init_sigma, init_rho):
        super().__init__()
        self.pd_algo = PD_Standalone_algo()
        
        self.tau_param = nn.Parameter(tc.tensor(init_tau))
        self.sigma_param = nn.Parameter(tc.tensor(init_sigma))
        self.rho_param = nn.Parameter(tc.tensor(init_rho))
        self.softplus = nn.Softplus()

    def forward(self, static, dynamic, y):
        Hmat = static
        p, p_old, d, d_old = dynamic
        
        tau = self.softplus(self.tau_param)
        sigma = self.softplus(self.sigma_param)
        rho = self.softplus(self.rho_param)
        
        p_new, d_new = self.pd_algo.iter_PD(p, p_old, d, d_old, y, Hmat, tau, sigma, rho)
        return p_new, d_new, (tau, sigma)

class PD_Standalone_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = PD_Standalone_algo()

        for _ in range(num_layers):
            self.Layers.append(PD_Standalone_layer(0.01, 0.01, 0.1))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        if static is None:
            Hmat, p0, d0 = self.algo.init_PD(x0, y)
            static = Hmat
            # dynamic stocke les états courants et précédents
            dynamic = (p0, p0, d0, d0)
            
        Hmat = static
        p, p_old, d, d_old = dynamic
        
        learned_params = []

        for layer in self.Layers:
            p_new, d_new, params = layer(static, (p, p_old, d, d_old), y)
            p_old = p
            d_old = d
            p = p_new
            d = d_new
            learned_params.append(params)
        
        dynamic_new = (p, p_old, d, d_old)
        return p, dynamic_new, learned_params