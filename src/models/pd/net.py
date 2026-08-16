import torch as tc
import torch.nn as nn
from src.models.pd.algo import PD_Standalone_algo

class PD_Standalone_layer(nn.Module):
    def __init__(self, init_tau, init_sigma, init_rho):
        super().__init__()
        self.pd_algo = PD_Standalone_algo()

        # tau_param/sigma_param sont maintenant des logits (sigmoid -> (0,1))
        # utilises comme fractions de la borne de stabilite de Chambolle-Pock,
        # et non plus des valeurs brutes passees dans un softplus. Voir
        # forward() pour la reparametrisation garantissant tau*sigma*||H||^2 <= 1.
        self.tau_param = nn.Parameter(tc.tensor(init_tau))
        self.sigma_param = nn.Parameter(tc.tensor(init_sigma))
        self.rho_param = nn.Parameter(tc.tensor(init_rho))
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, static, dynamic, y, L2):
        """
        Args:
            L2 : carre de la norme d'operateur (plus grande valeur singuliere
                 au carre) de Hmat, transmis par PD_Standalone_model.forward.
                 Sert a reparametrer tau/sigma pour garantir la convergence
                 du schema de Chambolle-Pock (condition tau*sigma*||H||^2 <= 1).
        """
        Hmat = static
        p, p_old, d, d_old = dynamic

        # Reparametrisation stable : tau, sigma in (0, 1/sqrt(L2)) chacun,
        # via des sigmoides independantes. Le produit
        # tau * sigma * L2 = sigmoid(tau_param) * sigmoid(sigma_param) < 1
        # est donc garanti par construction, quelle que soit la valeur des
        # parametres appris (plus de divergence numerique possible).
        inv_sqrt_L2 = 1.0 / tc.sqrt(L2 + 1e-12)
        tau = self.sigmoid(self.tau_param) * inv_sqrt_L2
        sigma = self.sigmoid(self.sigma_param) * inv_sqrt_L2
        rho = self.softplus(self.rho_param)

        p_new, d_new = self.pd_algo.iter_PD(p, p_old, d, d_old, y, Hmat, tau, sigma, rho)
        return p_new, d_new, (tau, sigma)


class PD_Standalone_model(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.Layers = nn.ModuleList()
        self.num_layers = num_layers
        self.algo = PD_Standalone_algo()

        for _ in range(num_layers):
            self.Layers.append(PD_Standalone_layer(0.01, 0.01, 0.1))

    def forward(self, static, dynamic, x0, y, x_true=None, lmbd_override=None, tau_override=None):
        if static is None:
            Hmat, p0, d0 = self.algo.init_PD(x0, y)
            static = Hmat
            # dynamic stocke les états courants et précédents
            dynamic = (p0, p0, d0, d0)

        Hmat = static
        p, p_old, d, d_old = dynamic

        # Carre de la norme d'operateur de Hmat (borne de stabilite de
        # Chambolle-Pock : tau*sigma*||H||^2 <= 1). Calcule une seule fois
        # par appel forward (Hmat est statique pour un couple (N, M) donne).
        # tc.linalg.matrix_norm(..., ord=2) renvoie la plus grande valeur
        # singuliere ; on ne retro-propage pas au travers de ce calcul
        # (Hmat n'est pas un parametre appris).
        with tc.no_grad():
            L2 = tc.linalg.matrix_norm(Hmat, ord=2) ** 2

        learned_params = []

        for layer in self.Layers:
            p_new, d_new, params = layer(static, (p, p_old, d, d_old), y, L2)
            p_old = p
            d_old = d
            p = p_new
            d = d_new
            learned_params.append(params)
        
        dynamic_new = (p, p_old, d, d_old)
        return p, dynamic_new, learned_params

