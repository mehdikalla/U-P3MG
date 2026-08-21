import torch
import torch.nn as nn
from src.models.p3mg.primal_dual.algo import PrimalDual_algo  
S = nn.Softplus()

# Primal-Dual model layers
class PD_layer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.pd_algo = PrimalDual_algo()

    def forward(self, sub_static, w_new, tau_scalar):
        w_new = self.pd_algo.iter_PD(sub_static, w_new, tau_scalar)
        return w_new

class PD_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList([PD_layer() for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, sub_static, w0, tau_params):
        w = w0
        for j in range(self.num_layers):
            tau_j = tau_params[j]
            l = self.Layers[j]
            w = l(sub_static, w, tau_j)
        return w
# -------------------