import torch.nn as nn
import torch as tc
from src.models.pmms.algo import PMMS_algo

# Valeur fixe de reference pour l'hyperparametre nu, utilisee a chaque couche.
DEFAULT_NU = 8.0e-5


class PMMS_layer(nn.Module):
    """
    Couche d'unrolling PMMS non entrainable.

    L'hyperparametre nu est fixe (DEFAULT_NU) a chaque couche, sans reseau
    de correction ni parametre appris.
    """

    def __init__(self):
        super().__init__()
        self.pmms_algo = PMMS_algo()

    def forward(self, static, dynamic, x, y, nu_override=None):
        if nu_override is not None:
            nu = nu_override.to(x.device).double()
        else:
            nu = tc.tensor(DEFAULT_NU, dtype=x.dtype, device=x.device)

        with tc.no_grad():
            x_new, dynamic_new = self.pmms_algo.iter_PMMS(static, dynamic, x, y, nu)

        return x_new, dynamic_new, nu


class PMMS_model(nn.Module):
    """
    Modele d'unrolling PMMS non entraine.

    Le reseau est constitue de la repetition de `num_layers` iterations de
    l'algorithme PMMS, avec nu fixe a DEFAULT_NU (aucun parametre appris,
    aucun gradient calcule).
    """

    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = PMMS_algo()

        for _ in range(num_layers):
            self.Layers.append(PMMS_layer())

        # Aucun parametre entrainable : reseau fige.
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, static, dynamic, x0, y, x_true=None, nu_override=None):
        if static is None:
            static, dynamic = self.algo.init_PMMS(x0, y)

        x = x0

        dynamic_nu = []

        for layer in self.Layers:
            x, dynamic, nu_k = layer(static, dynamic, x, y, nu_override)
        dynamic_nu.append(nu_k)

        return x, dynamic, dynamic_nu
