import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat


class PD_Standalone_algo(nn.Module):
    """
    Algorithme Primal-Dual autonome (schema de Chambolle-Pock), aligne sur
    l'architecture du module Primal-Dual interne de P3MG
    (src/models/p3mg/primal_dual/algo.py) : une methode init_PD qui prepare
    l'etat initial (w0) et les quantites statiques (sub_static), et une
    methode iter_PD qui effectue une iteration a partir de ces deux objets
    et d'un unique hyperparametre appris/recherche (tau).
    """

    def __init__(self, margin: float = 0.99):
        super().__init__()
        # Marge de securite sous la borne critique de stabilite du schema de
        # Chambolle-Pock (tau * sigma * ||H||^2 <= 1).
        self.margin = margin

    def init_PD(self, x0, y):
        """
        Initialisation des variables et calculs preliminaires pour le
        primal-dual.

        Retourne :
            w0 = [p0, d0] : etat initial (primal, dual)
            sub_static = [Hmat, L2] : quantites statiques (matrice d'observation
                et carre de sa norme d'operateur), reutilisees a chaque iteration.
        """
        P, N, M = x0.size(0), x0.size(1), y.size(1)

        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        # Initialisation du primal (p) et du dual (d)
        p0 = x0
        d0 = tc.zeros_like(tc.matmul(x0, Hmat.t()))

        # Carre de la norme d'operateur de Hmat (borne de stabilite de
        # Chambolle-Pock). Calcule une seule fois, sans retropropagation
        # (Hmat n'est pas un parametre appris).
        with tc.no_grad():
            L2 = tc.linalg.matrix_norm(Hmat, ord=2) ** 2

        w0 = [p0, d0]
        sub_static = [Hmat, L2]

        return w0, sub_static

    def iter_PD(self, sub_static, w, y, tau):
        """
        Iteration Primal-Dual autonome (schema de Chambolle-Pock).

        Ordre de l'algorithme :
        1. Mise a jour du primal a partir du dual courant (non extrapole) :
           p_new = prox_primal(p - tau * H^T d)
        2. Extrapolation du primal (over-relaxation, theta = 1) :
           p_bar = 2 * p_new - p
        3. Mise a jour du dual a partir du primal extrapole :
           d_new = prox_dual(d + sigma * H * p_bar)

        Seul tau est un hyperparametre libre. sigma est reparametre de
        maniere deterministe a partir de tau afin de garantir par
        construction la condition de convergence tau * sigma * ||H||^2 <= 1
        (voir self.margin). Le terme de regularisation duale (rho) est fixe
        a 0, ce qui correspond au prox usuel de l'attache aux donnees L2
        ||Hx - y||^2 sans terme additionnel.
        """
        Hmat, L2 = sub_static
        p, d = w

        sigma = self.margin / (tau * L2 + 1e-12)

        # 1. Mise a jour du primal a partir du dual courant
        bp = -tau * tc.matmul(d, Hmat)
        p_new = tc.nn.functional.relu(p + bp)  # Activation primal (ex: ReLU)

        # 2. Extrapolation du primal (over-relaxation)
        p_bar = 2 * p_new - p

        # 3. Mise a jour du dual a partir du primal extrapole
        bd = sigma * tc.matmul(p_bar, Hmat.t())
        d_new = d + bd - sigma * y

        w_new = [p_new, d_new]
        return w_new
