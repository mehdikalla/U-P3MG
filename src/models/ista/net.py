import torch as tc
import torch.nn as nn
from src.models.ista.algo import ISTA_algo

class ISTA_layer(nn.Module):
    def __init__(self, initial_gamma, initial_lmbd):
        super().__init__()
        self.ista_algo = ISTA_algo()
        
        # Paramètres apprenables par couche (positifs via Softplus)
        self.gamma_param = nn.Parameter(tc.tensor(initial_gamma))
        self.lmbd_param = nn.Parameter(tc.tensor(initial_lmbd))
        self.softplus = nn.Softplus()

    def forward(self, Hmat, x, y, L=None):
        gamma = self.softplus(self.gamma_param)
        lmbd = self.softplus(self.lmbd_param)
        x_new = self.ista_algo.iter_ISTA(x, y, Hmat, gamma, lmbd)
        return x_new, lmbd

class ISTA_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers=None):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = ISTA_algo() 

        # Valeurs initiales par défaut
        init_gamma = 0.001 
        init_lmbd = 0.1

        for _ in range(num_layers):
            self.Layers.append(ISTA_layer(init_gamma, init_lmbd))
        
        self._params_initialized = False

    def init_params_from_static(self, static):
        """
        Initialise gamma <= 1/L pour assurer la convergence au démarrage.
        """
        _, L = static
        if L == 0: return # Sécurité
        
        gamma_init = 1.0 / L.item()
        
        # Inversion du Softplus: x = ln(exp(y) - 1)
        import math
        # On ajoute une petite marge de sécurité (0.95/L)
        val = 0.95 * gamma_init
        inv_softplus_gamma = math.log(math.exp(val) - 1)
        
        for layer in self.Layers:
            with tc.no_grad():
                layer.gamma_param.fill_(inv_softplus_gamma)

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        x = x0
        
        # Initialisation statique si nécessaire (Hmat, L)
        if static is None:
            static = self.algo.init_ISTA(x0, y)
            
            # On initialise les poids du réseau selon la physique (L) au premier passage
            if not self._params_initialized:
                 self.init_params_from_static(static)
                 self._params_initialized = True

        Hmat, L = static
        
        learned_lambdas = []

        for layer in self.Layers:
            x, lmbd_val = layer(Hmat, x, y, L=L)
            learned_lambdas.append(lmbd_val)
        
        # Retour compatible avec la signature P3MG (x, dynamic, list_lambdas)
        return x, None, learned_lambdas