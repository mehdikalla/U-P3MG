import torch as tc
import torch.nn as nn
from src.models.hq.algo import HQ_algo

class HQ_layer(nn.Module):
    def __init__(self, in_features=100):
        super().__init__()
        self.hq_algo = HQ_algo()
        
        self.fc_cvx = nn.Linear(in_features, 1, bias=True).double()
        self.fc_ncvx = nn.Linear(in_features, 1, bias=True).double()
        
        nn.init.uniform_(self.fc_cvx.weight, a=0.01, b=0.02)
        nn.init.constant_(self.fc_cvx.bias, 0.1) 
        
        nn.init.uniform_(self.fc_ncvx.weight, a=0.01, b=0.02)
        nn.init.constant_(self.fc_ncvx.bias, 0.1)
        
        self.gamma = nn.Parameter(tc.tensor([0.5], dtype=tc.float64), requires_grad=True)
        self.softplus = nn.Softplus()

    def forward(self, static, x, y, gamma_override=None, lmbd_cvx_override=None, lmbd_ncvx_override=None):
        # Ht_y a été supprimé des variables statiques
        Hmat, Ht_H = static
        
        Hx = tc.matmul(x, Hmat.t().contiguous())
        res = ((Hx - y) ** 2).contiguous()
        
        if gamma_override is not None:
            gamma_val = tc.tensor(gamma_override, device=x.device, dtype=tc.float64)
        else:
            gamma_val = self.softplus(self.gamma)
            
        if lmbd_cvx_override is not None:
            lmbd_cvx = tc.tensor(lmbd_cvx_override, device=x.device, dtype=tc.float64)
        else:
            lmbd_cvx = self.softplus(self.fc_cvx(res))
            
        if lmbd_ncvx_override is not None:
            lmbd_ncvx = tc.tensor(lmbd_ncvx_override, device=x.device, dtype=tc.float64)
        else:
            lmbd_ncvx = self.softplus(self.fc_ncvx(res))
        
        x_new = self.hq_algo.iter_HQ(x, y, Hmat, Ht_H, gamma_val, lmbd_cvx, lmbd_ncvx)
        return x_new, (lmbd_cvx, lmbd_ncvx, gamma_val)

class HQ_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers=None, M_dim=100, **kwargs):
        super().__init__()
        self.Layers = nn.ModuleList([HQ_layer(in_features=M_dim) for _ in range(num_layers)])
        self.algo = HQ_algo()

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_cvx_override=None, lmbd_ncvx_override=None, gamma_override=None):
        x = x0
        if static is None: static = self.algo.init_HQ(x0, y)
        learned_params = []
        for layer in self.Layers:
            x, params = layer(static, x, y, gamma_override, lmbd_cvx_override, lmbd_ncvx_override)
            learned_params.append(params)
        return x, None, learned_params