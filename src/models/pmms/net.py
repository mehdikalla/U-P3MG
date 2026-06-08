import torch.nn as nn
from src.models.pmms.algo import PMMS_algo

class PMMS_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pmms_algo = PMMS_algo()

    def forward(self, static, dynamic, x, y):
        x_new, dynamic_new = self.pmms_algo.iter_PMMS(static, dynamic, x, y)
        return x_new, dynamic_new

class PMMS_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = PMMS_algo()

        for _ in range(num_layers):
            self.Layers.append(PMMS_layer())

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        if static is None:
            static, dynamic = self.algo.init_PMMS(x0, y)
            
        x = x0
        
        for layer in self.Layers:
            x, dynamic = layer(static, dynamic, x, y)
            
        # L'algorithme PMMS n'apprend pas de paramètre par couche par défaut
        learned_lambdas = []
        return x, dynamic, learned_lambdas