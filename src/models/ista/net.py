import math
import torch as tc
import torch.nn as nn
from src.models.ista.algo import ISTA_algo


class ISTA_layer(nn.Module):
    """
    Une couche d'unrolling ISTA. Le pas `gamma` est parametre par une
    sigmoide bornee dans (0, 1/L) plutot qu'un softplus non borne, afin de
    garantir la stabilite theorique du schema de gradient proximal.
    """

    def __init__(self, initial_gamma_ratio, initial_lmbd):
        super().__init__()
        self.ista_algo = ISTA_algo()

        # Parametre brut de gamma, transforme par sigmoide vers le ratio
        # gamma/L_max souhaite. inv_sigmoid(p) = ln(p / (1 - p))
        init_ratio = min(max(initial_gamma_ratio, 1e-4), 1.0 - 1e-4)
        inv_sigmoid_gamma = math.log(init_ratio / (1.0 - init_ratio))
        self.gamma_param = nn.Parameter(tc.tensor(inv_sigmoid_gamma))

        # lambda reste parametre par softplus (positif, non borne
        # superieurement : aucune contrainte theorique similaire a gamma).
        self.lmbd_param = nn.Parameter(tc.tensor(initial_lmbd))
        self.softplus = nn.Softplus()

    def forward(self, Hmat, L, x, y):
        # gamma est garanti dans (0, 0.99/L) par construction, quelle que
        # soit la valeur apprise de gamma_param.
        gamma = 0.99 * tc.sigmoid(self.gamma_param) / L
        lmbd = self.softplus(self.lmbd_param)

        x_new = self.ista_algo.iter_ISTA(x, y, Hmat, gamma, lmbd)
        return x_new, lmbd


class ISTA_model(nn.Module):
    """
    Modele d'unrolling ISTA avec acceleration FISTA : gamma borne (cf.
    ISTA_layer) et extrapolation de Nesterov entre couches (convergence
    O(1/k^2) au lieu de O(1/k)) pour compenser le budget d'iterations reduit.
    """

    def __init__(self, num_layers, num_pd_layers=None):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = ISTA_algo()

        # Valeurs initiales par defaut (utilisees tant que `static` n'a
        # pas ete calcule au premier forward, cf. init_params_from_static).
        init_gamma_ratio = 0.95
        init_lmbd = 0.1

        for _ in range(num_layers):
            self.Layers.append(ISTA_layer(init_gamma_ratio, init_lmbd))

        self._params_initialized = False

    def init_params_from_static(self, static, args=None):
        """
        Initialise lambda a partir des bornes de recherche du config
        (lmbd_ist_min/lmbd_ist_max) plutot qu'une constante arbitraire,
        pour eviter de saturer l'operateur de seuillage doux des le depart.
        """
        Hmat, L = static
        if L == 0:
            return

        lmbd_min = getattr(args, 'lmbd_ist_min', 1e-9) if args is not None else 1e-9
        lmbd_max = getattr(args, 'lmbd_ist_max', 1.1) if args is not None else 1.1
        lmbd_min = max(float(lmbd_min), 1e-12)
        lmbd_max = max(float(lmbd_max), lmbd_min * 10)
        # Moyenne geometrique des bornes : point de depart neutre dans
        # l'espace de recherche log-uniforme utilise par le random search.
        lmbd_init = math.sqrt(lmbd_min * lmbd_max)
        inv_softplus_lmbd = math.log(math.expm1(lmbd_init)) if lmbd_init > 1e-6 else math.log(lmbd_init)

        for layer in self.Layers:
            with tc.no_grad():
                layer.lmbd_param.fill_(inv_softplus_lmbd)

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None, args=None):
        # Initialisation statique si necessaire (Hmat, L)
        if static is None:
            static = self.algo.init_ISTA(x0, y)

            # On initialise les poids du reseau selon la physique (L) et
            # les bornes de recherche de lambda au premier passage.
            if not self._params_initialized:
                self.init_params_from_static(static, args=args)
                self._params_initialized = True

        Hmat, L = static

        learned_lambdas = []

        # Schema FISTA : x = iteres, z = point extrapole, t = poids de Nesterov.
        x_prev = x0
        z = x0
        t = 1.0

        for layer in self.Layers:
            x_new, lmbd_val = layer(Hmat, L, z, y)
            learned_lambdas.append(lmbd_val)

            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x_prev)

            x_prev = x_new
            t = t_new

        x = x_prev

        # Retour compatible avec la signature P3MG (x, dynamic, list_lambdas)
        return x, None, learned_lambdas