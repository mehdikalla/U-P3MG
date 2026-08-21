import torch as tc
import torch.nn as nn
from src.models.pd.algo import PD_Standalone_algo

S = nn.Softplus()

# PD Standalone model layers
class PD_layer(nn.Module):
    """
    Couche Primal-Dual autonome (cf. src/models/p3mg/primal_dual/net.py) :
    pas de descente (tau) fixe, lambda_reg transmis directement a iter_PD.
    """

    def __init__(self):
        super().__init__()
        self.pd_algo = PD_Standalone_algo()

    def forward(self, sub_static, w_new, y, tau_scalar, lambda_scalar):
        w_new = self.pd_algo.iter_PD(sub_static, w_new, tau_scalar, lambda_scalar)
        return w_new


class PD_model(nn.Module):
    """
    Modele Primal-Dual autonome deroule (cf. PD_model dans
    src/models/p3mg/primal_dual/net.py). tau_params est fixe (pas de
    Chambolle-Pock, sans effet a convergence) ; lambda_params est l'unique
    parametre appris, ponderant la regularisation quadratique.
    """

    def __init__(self, num_layers, tau_fixed: float = 1.0):
        super().__init__()
        self.Layers = nn.ModuleList([PD_layer() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.algo = PD_Standalone_algo()

        # tau_fixed : coefficient de relaxation fixe, non appris (cf. iter_PD).
        self.register_buffer('tau_fixed', tc.tensor(tau_fixed).double())

        # lambda_params : un logit par couche, transforme via softplus en
        # lambda_reg positif. Seul hyperparametre appris du modele.
        self.lambda_params = nn.Parameter(tc.empty(num_layers).double().fill_(0.0))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        if static is None:
            w_new, sub_static = self.algo.init_PD(x0, y)
            static = sub_static
            dynamic = w_new

        sub_static = static
        w_new = dynamic

        if lmbd_override is not None:
            lmbd_val = lmbd_override.to(x0.device).double()
            lambda_params = lmbd_val.expand(self.lambda_params.shape)
        else:
            lambda_params = S(self.lambda_params)

        if tau_override is not None:
            tau_val = tau_override.to(x0.device).double()
            tau_params = tau_val.expand(self.num_layers)
        else:
            tau_params = self.tau_fixed.expand(self.num_layers)

        learned_params = []
        for j, layer in enumerate(self.Layers):
            tau_j = tau_params[j]
            lambda_j = lambda_params[j]
            w_new = layer(sub_static, w_new, y, tau_j, lambda_j)
            learned_params.append(lambda_j)

        un, vn = w_new
        dynamic_new = w_new
        return un, dynamic_new, learned_params


# Alias retrocompatibles : le modele conserve son nom historique
# 'PD_Standalone_layer' / 'PD_Standalone_model' pour les imports existants.
PD_Standalone_layer = PD_layer
PD_Standalone_model = PD_model
