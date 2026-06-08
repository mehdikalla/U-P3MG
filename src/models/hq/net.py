import torch as tc
import torch.nn as nn
from src.models.hq.algo import HQ_algo

class HQ_layer(nn.Module):
    def __init__(self, initial_gamma, initial_lmbd_cvx, initial_lmbd_ncvx):
        super().__init__()
        self.hq_algo = HQ_algo()
        
        # Paramètres optimisables
        self.gamma_param = nn.Parameter(tc.tensor(initial_gamma))
        self.lmbd_cvx_param = nn.Parameter(tc.tensor(initial_lmbd_cvx))
        self.lmbd_ncvx_param = nn.Parameter(tc.tensor(initial_lmbd_ncvx))
        self.softplus = nn.Softplus()

    def forward(self, static, x, y):
        # Récupération des 3 variables précalculées
        Hmat, Ht_y, Ht_H = static
        
        # Application de la contrainte de positivité
        gamma = self.softplus(self.gamma_param)
        lmbd_cvx = self.softplus(self.lmbd_cvx_param)
        lmbd_ncvx = self.softplus(self.lmbd_ncvx_param)
        
        x_new = self.hq_algo.iter_HQ(x, y, Hmat, Ht_y, Ht_H, gamma, lmbd_cvx, lmbd_ncvx)
        
        return x_new, (lmbd_cvx, lmbd_ncvx)

class HQ_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers=None):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = HQ_algo()

        # Initialisation conventionnelle
        init_gamma = 0.001 
        init_lmbd_cvx = 0.1
        init_lmbd_ncvx = 0.1

        for _ in range(num_layers):
            self.Layers.append(HQ_layer(init_gamma, init_lmbd_cvx, init_lmbd_ncvx))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        x = x0
        
        if static is None:
            static = self.algo.init_HQ(x0, y)

        learned_lambdas = []

        for layer in self.Layers:
            x, lmbds = layer(static, x, y)
            learned_lambdas.append(lmbds)
        
        return x, None, learned_lambdas