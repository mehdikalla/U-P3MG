import torch.nn as nn
import torch as tc
from src.models.pmms.algo import PMMS_algo
from src.models.FC_block import FC_block
S = nn.Softplus()

class PMMS_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pmms_algo = PMMS_algo()
        self.f_act = FC_block([100, 50, 25, 12, 1])

    def forward(self, static, dynamic, x, y, nu_override=None):
        # Gestion Nu
        if nu_override is not None:
            nu = nu_override.to(x.device).double()
        else:
            nu = S(self.f_act(tc.pow(y,2)))
        

        x_new, dynamic_new = self.pmms_algo.iter_PMMS(static, dynamic, x, y, nu)
        return x_new, dynamic_new, nu

class PMMS_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = PMMS_algo()

        for _ in range(num_layers):
            self.Layers.append(PMMS_layer())

    def forward(self, static, dynamic, x0, y, x_true=None, nu_override=None):
        if static is None:
            static, dynamic = self.algo.init_PMMS(x0, y)
            
        x = x0

        dynamic_nu = []

        for layer in self.Layers:
            x, dynamic, nu_k = layer(static, dynamic, x, y, nu_override)
        dynamic_nu.append(nu_k)

        return x, dynamic, dynamic_nu