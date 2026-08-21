import math
import torch.nn as nn
import torch as tc
from src.models.pmms.algo import PMMS_algo

# Valeur fixe de reference pour l'hyperparametre nu, utilisee comme point
# de depart de l'apprentissage (cf. PMMS_model.init_params_from_args).
DEFAULT_NU = 8.0e-5


class PMMS_layer(nn.Module):
    """
    Couche d'unrolling PMMS avec hyperparametre nu appris par couche.

    Contrairement a la version precedente (nu fixe a DEFAULT_NU, forward
    execute sous torch.no_grad(), tous les parametres geles), nu est ici
    un parametre reel du reseau, positif par construction via Softplus.
    Sans cette correction, le "reseau deroule" n'etait qu'une repetition
    de l'algorithme PMMS classique avec un hyperparametre non calibre, ce
    qui explique la sous-performance systematique face a l'equivalent
    random_search (qui calibre nu par recherche log-uniforme sur
    n_trials tirages et l'applique ensuite sur algo_iters iterations).
    """

    def __init__(self, initial_nu):
        super().__init__()
        self.pmms_algo = PMMS_algo()
        self.softplus = nn.Softplus()

        # Inversion du softplus : nu_param tel que softplus(nu_param) = initial_nu
        init_nu = max(float(initial_nu), 1e-12)
        inv_softplus_nu = math.log(math.expm1(init_nu)) if init_nu > 1e-6 else math.log(init_nu)
        self.nu_param = nn.Parameter(tc.tensor(inv_softplus_nu))

    def forward(self, static, dynamic, x, y, nu_override=None):
        if nu_override is not None:
            nu = nu_override.to(x.device).double()
        else:
            nu = self.softplus(self.nu_param).to(dtype=x.dtype, device=x.device)

        x_new, dynamic_new = self.pmms_algo.iter_PMMS(static, dynamic, x, y, nu)

        return x_new, dynamic_new, nu


class PMMS_model(nn.Module):
    """
    Modele d'unrolling PMMS avec nu appris independamment a chaque couche.

    Chaque couche dispose de son propre parametre nu (positif via
    Softplus), initialise a DEFAULT_NU (valeur de reference physique) puis
    affine par retropropagation sur la loss de reconstruction, a la
    maniere du lambda appris par couche dans le modele ISTA. Ceci
    remplace l'ancienne version figee (nu constant, reseau gele) qui ne
    beneficiait d'aucun apprentissage.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = PMMS_algo()

        for _ in range(num_layers):
            self.Layers.append(PMMS_layer(DEFAULT_NU))

    def forward(self, static, dynamic, x0, y, x_true=None, nu_override=None):
        # static et dynamic doivent toujours etre initialises ensemble : si l'un
        # manque, on reinitialise les deux via sigma/beta/eta si disponibles.
        if dynamic is None:
            if static is not None:
                _, sigma, beta, eta, _ = static
                static, dynamic = self.algo.init_PMMS(x0, y, sigma=sigma, beta=beta, eta=eta)
            else:
                static, dynamic = self.algo.init_PMMS(x0, y)

        x = x0

        dynamic_nu = []

        for layer in self.Layers:
            x, dynamic, nu_k = layer(static, dynamic, x, y, nu_override)
            dynamic_nu.append(nu_k)

        return x, dynamic, dynamic_nu

