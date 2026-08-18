import torch as tc
import torch.nn as nn
from src.models.pd.algo import PD_Standalone_algo

S = nn.Softplus()

# -------------------------
# PD Standalone model layers
# -------------------------
class PD_Standalone_layer(nn.Module):
    """
    Couche Primal-Dual autonome, alignee sur PD_layer
    (src/models/p3mg/primal_dual/net.py) : un unique hyperparametre appris,
    tau, transmis directement a iter_PD.
    """

    def __init__(self):
        super().__init__()
        self.pd_algo = PD_Standalone_algo()

    def forward(self, sub_static, w, y, tau_scalar):
        w_new = self.pd_algo.iter_PD(sub_static, w, y, tau_scalar)
        return w_new


class PD_Standalone_model(nn.Module):
    """
    Modele Primal-Dual autonome deroule, aligne sur PD_model
    (src/models/p3mg/primal_dual/net.py) : une seule sequence de couches
    partageant le meme algorithme, chacune parametree par un scalaire tau_j
    appris independamment (tau_params).
    """

    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList([PD_Standalone_layer() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.algo = PD_Standalone_algo()

        # tau_params : un logit par couche, transforme via sigmoid en (0, 1)
        # puis reparametre dans iter_PD en fraction de la borne de stabilite
        # de Chambolle-Pock. Seul hyperparametre appris du modele.
        self.tau_params = nn.Parameter(tc.empty(num_layers).double().fill_(0.01))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        if static is None:
            w0, sub_static = self.algo.init_PD(x0, y)
            static = sub_static
            dynamic = w0

        sub_static = static
        w = dynamic

        if tau_override is not None:
            tau_val = tau_override.to(x0.device).double()
            tau_params = tau_val.expand(self.tau_params.shape)
        else:
            tau_params = tc.sigmoid(self.tau_params)

        learned_params = []
        for j, layer in enumerate(self.Layers):
            tau_j = tau_params[j]
            w = layer(sub_static, w, y, tau_j)
            learned_params.append(tau_j)

        p, d = w
        dynamic_new = w
        return p, dynamic_new, learned_params
