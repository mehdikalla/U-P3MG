import math
import torch as tc
import torch.nn as nn
from src.models.p3mg.algo import P3MG_algo
from src.models.FC_block import FC_block
S = nn.Softplus()

# Valeur d'initialisation par defaut pour lambda lorsqu'il est entraine
# directement comme un nn.Parameter (mode use_fc_lambda=False).
LMBD_PARAM_INIT = 8e-5

def S2(x):
    return 2*tc.sigmoid(x)

def _inv_softplus(y: float) -> float:
    """Inverse de Softplus : renvoie x tel que softplus(x) = y (y > 0)."""
    return math.log(math.expm1(y))

# --------------------
# P3MG model layers
# --------------------
class layer_0(nn.Module):
    def __init__(self, num_pd_layers: int, mlp_hidden: list = None, use_fc_lambda: bool = True):
        super().__init__()
        self.p3mg_algo = P3MG_algo(num_pd_layers)
        self.use_fc_lambda = use_fc_lambda

        if self.use_fc_lambda:
            # 1. Lambda (Dynamique) - hidden layers du MLP configurables (ablation c)
            hidden = mlp_hidden if mlp_hidden else [50, 25, 12]
            self.f_act = FC_block([100, *hidden, 1])
        else:
            # 1bis. Lambda (Statique) - scalaire appris directement, comme tau_k,
            # initialise a LMBD_PARAM_INIT via l'inverse de Softplus.
            init_raw = _inv_softplus(LMBD_PARAM_INIT)
            self.lmbd_raw = nn.Parameter(tc.tensor(init_raw).double(), requires_grad=True)

        # 2. Tau (Explicite - Vecteur de M valeurs)
        self.tau_k = nn.Parameter(tc.empty(num_pd_layers).double().fill_(0.5), requires_grad=True) 

    def forward(self, static, dynamic, x, y, lmbd_override=None, tau_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        x = x.to(device)
        y = y.to(device)
        
        # Gestion Lambda
        if lmbd_override is not None:
             lmbd = lmbd_override.to(x.device).double()
        elif self.use_fc_lambda:
             lmbd = S(self.f_act(tc.pow(y,2)))
        else:
             lmbd = S(self.lmbd_raw)
        
        # Gestion Tau (Override ou Appris)
        if tau_override is not None:
             # Si override (scalaire), on l'étend pour correspondre au nombre de sous-couches
             tau_val = tau_override.to(x.device).double()
             tau_params = tau_val.expand(self.tau_k.shape)
        else:
             tau_params = S2(self.tau_k) 

        x_new, dynamic_new = self.p3mg_algo.iter_P3MG_base(static, x, y, lmbd, tau_params)
        return x_new, dynamic_new, lmbd


class layer_k(nn.Module):
    def __init__(self, num_pd_layers: int, mlp_hidden: list = None, use_fc_lambda: bool = True):
        super().__init__()
        self.p3mg_algo = P3MG_algo(num_pd_layers)
        self.use_fc_lambda = use_fc_lambda

        if self.use_fc_lambda:
            hidden = mlp_hidden if mlp_hidden else [50, 25, 12]
            self.f_act = FC_block([100, *hidden, 1])
        else:
            init_raw = _inv_softplus(LMBD_PARAM_INIT)
            self.lmbd_raw = nn.Parameter(tc.tensor(init_raw).double(), requires_grad=True)

        self.tau_k = nn.Parameter(tc.empty(num_pd_layers).double().fill_(0.5), requires_grad=True)

    def forward(self, static, dynamic, x, y, lmbd_override=None, tau_override=None):
        device = 'cuda' if tc.cuda.is_available() else 'cpu'
        x = x.to(device)
        y = y.to(device)
        
        if lmbd_override is not None:
             lmbd = lmbd_override.to(x.device).double()
        elif self.use_fc_lambda:
             lmbd = S(self.f_act(tc.pow(y,2)))
        else:
             lmbd = S(self.lmbd_raw)
             
        if tau_override is not None:
             tau_val = tau_override.to(x.device).double()
             tau_params = tau_val.expand(self.tau_k.shape)
        else:
             tau_params = S2(self.tau_k)
            
        x_new, dynamic_new = self.p3mg_algo.iter_P3MG(static, dynamic, x, y, lmbd, tau_params)
        return x_new, dynamic_new, lmbd


# --------------------
# P3MG model container
# --------------------
class P3MG_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers, mlp_hidden: list = None, use_fc_lambda: bool = True):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.num_pd_layers = num_pd_layers
        self.mlp_hidden = mlp_hidden
        self.use_fc_lambda = use_fc_lambda

        for i in range(num_layers):
            if i == 0: self.Layers.append(layer_0(num_pd_layers, mlp_hidden, use_fc_lambda))
            else: self.Layers.append(layer_k(num_pd_layers, mlp_hidden, use_fc_lambda))


    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        x, dyn = x0, dynamic
        dynamic_lambdas = []
        
        for l in self.Layers:
            # On passe les deux overrides
            x, dyn, lmbd_k = l(static, dyn, x, y, lmbd_override=lmbd_override, tau_override=tau_override) 
            dynamic_lambdas.append(lmbd_k)
        
        return x, dyn, dynamic_lambdas