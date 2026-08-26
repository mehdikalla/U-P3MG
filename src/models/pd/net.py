import torch as tc
import torch.nn as nn
from src.models.pd.algo import PD_Standalone_algo

S = nn.Softplus()

# PD Standalone model layers
class PD_layer(nn.Module):
    """
    Couche Primal-Dual autonome (
    """

    def __init__(self, tau_margin: float = 0.99):
        super().__init__()
        self.pd_algo = PD_Standalone_algo()
        self.tau_margin = tau_margin
        self.tau_raw = nn.Parameter(tc.tensor(0.0).double())

    def forward(self, sub_static, w_new, y, tau_override, lambda_scalar):
        if tau_override is not None:
            tau_scalar = tau_override
        else:
            tau_scalar = self.tau_margin * 2.0 * tc.sigmoid(self.tau_raw)
        w_new = self.pd_algo.iter_PD(sub_static, w_new, y, tau_scalar, lambda_scalar)
        return w_new, tau_scalar


class PD_model(nn.Module):
    """
    Modele Primal-Dual autonome deroule 
    """

    def __init__(self, num_layers, tau_fixed: float = 1.0):
        super().__init__()
        self.Layers = nn.ModuleList([PD_layer() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.algo = PD_Standalone_algo()

        # tau_fixed : conserve uniquement comme valeur de repli lorsque
        # tau_override est fourni explicitement (cf. random_search/compare).
        self.register_buffer('tau_fixed', tc.tensor(tau_fixed).double())

        # lambda_params : un logit par couche, transforme via softplus en
        # lambda_reg positif.
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
            tau_override_per_layer = tau_val.expand(self.num_layers)
        else:
            tau_override_per_layer = [None] * self.num_layers

        learned_params = []
        for j, layer in enumerate(self.Layers):
            lambda_j = lambda_params[j]
            w_new, tau_j = layer(sub_static, w_new, y, tau_override_per_layer[j], lambda_j)
            learned_params.append(lambda_j)

        un, vn = w_new
        dynamic_new = w_new
        return un, dynamic_new, learned_params


# Alias retrocompatibles : le modele conserve son nom historique
# 'PD_Standalone_layer' / 'PD_Standalone_model' pour les imports existants.
PD_Standalone_layer = PD_layer
PD_Standalone_model = PD_model

