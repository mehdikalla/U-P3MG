import torch as tc
import torch.nn as nn
from src.models.ista.algo import ISTA_algo

class ISTA_layer(nn.Module):
    """
    Couche ISTA deroulee, avec pas de descente `gamma` parametre de facon
    a garantir la stabilite de l'iteration quelle que soit la valeur prise
    par le parametre appris au cours de l'entrainement.

    """

    def __init__(self, initial_lmbd, gamma_margin: float = 0.99):
        super().__init__()
        self.ista_algo = ISTA_algo()
        self.gamma_margin = gamma_margin

        # Parametre brut controlant gamma via une sigmoide bornee
        self.gamma_raw = nn.Parameter(tc.tensor(0.0))

        # lmbd (poids de regularisation L1) reste positif via Softplus,
        # sans borne superieure (aucune contrainte de stabilite dessus).
        self.lmbd_param = nn.Parameter(tc.tensor(initial_lmbd))
        self.softplus = nn.Softplus()

    def forward(self, Hmat, L, x, y):
        # Pas de descente borne de facon inconditionnelle dans (0, margin*2/L).
        gamma = self.gamma_margin * tc.sigmoid(self.gamma_raw) * (2.0 / L)
        lmbd = self.softplus(self.lmbd_param)

        x_new = self.ista_algo.iter_ISTA(x, y, Hmat, gamma, lmbd)
        return x_new, lmbd, gamma

class ISTA_model(nn.Module):
    def __init__(self, num_layers, num_pd_layers=None):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = ISTA_algo()

        # Valeur initiale par defaut pour lambda (le pas gamma est
        # desormais parametre directement de facon stable, cf. ISTA_layer).
        init_lmbd = 0.1

        for _ in range(num_layers):
            self.Layers.append(ISTA_layer(init_lmbd))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        x = x0

        # Initialisation statique si necessaire (Hmat, L)
        if static is None:
            static = self.algo.init_ISTA(x0, y)

        Hmat, L = static

        learned_lambdas = []

        for layer in self.Layers:
            x, lmbd_val, _ = layer(Hmat, L, x, y)
            learned_lambdas.append(lmbd_val)

        # Retour compatible avec la signature P3MG (x, dynamic, list_lambdas)
        return x, None, learned_lambdas
