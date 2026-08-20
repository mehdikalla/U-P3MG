import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat, proj_simplex


class PD_Standalone_algo(nn.Module):
    """
    Algorithme Primal-Dual autonome (schema de Chambolle-Pock), aligne sur
    l'architecture du module Primal-Dual interne de P3MG
    (src/models/p3mg/primal_dual/algo.py) : une methode init_PD qui prepare
    l'etat initial (w0) et les quantites statiques (sub_static), et une
    methode iter_PD qui effectue une iteration a partir de ces deux objets.

    tau est desormais fixe (non recherche) : c'est un pur parametre de pas
    garantissant la stabilite du schema de Chambolle-Pock, sans influence
    sur le probleme d'optimisation resolu. L'unique hyperparametre recherche
    par le random search est lambda_reg, qui pondere un terme de
    regularisation quadratique reellement present dans le probleme resolu
    (voir iter_PD).
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

    def iter_PD(self, sub_static, w, y, tau, lambda_reg=0.0):
        """
        Iteration Primal-Dual autonome (schema de Chambolle-Pock) avec
        regularisation quadratique explicite.

        Probleme resolu :
            min_x  0.5 * ||H x - y||^2  +  0.5 * lambda_reg * ||x||^2
            s.c.   x dans le simplexe (positivite + somme = 1)

        Ordre de l'algorithme :
        1. Mise a jour du primal a partir du dual courant (non extrapole),
           avec retrecissement quadratique controle par lambda_reg puis
           projection sur le simplexe :
           p_new = proj_simplex( (p - tau * H^T d) / (1 + tau * lambda_reg) )
        2. Extrapolation du primal (over-relaxation, theta = 1) :
           p_bar = 2 * p_new - p
        3. Mise a jour du dual a partir du primal extrapole (prox exact de
           la conjuguee de l'attache aux donnees L2) :
           d_new = (d + sigma * H * p_bar - sigma * y) / (1 + sigma)

        tau est un parametre de pas fixe, transmis explicitement a chaque
        appel (non recherche) : il ne fait que garantir, via sigma qui en
        est deduit de maniere deterministe, la condition de stabilite
        tau * sigma * ||H||^2 <= 1 (voir self.margin). tau ne modifie pas le
        probleme resolu ci-dessus ; a convergence, sa valeur (tant qu'elle
        assure la stabilite) n'a donc aucun effet sur la loss finale, ce qui
        explique pourquoi la recherche aleatoire sur tau seul donnait une
        loss constante.

        lambda_reg est desormais l'unique hyperparametre recherche par le
        random search pour ce modele (cf. src/strategies/random_search.py).
        Contrairement a tau, il modifie reellement le probleme d'optimisation
        resolu (terme de regularisation quadratique), donc la loss de
        calibration varie effectivement avec lambda_reg, y compris a
        convergence complete (grand nombre d'iterations). Avec
        lambda_reg = 0, on retrouve exactement le schema de Chambolle-Pock
        standard, sans regularisation.

        Contraintes/prox alignes sur les autres modeles de reference
        (P3MG, PMMS, ISTA) :
        - Le prox primal est la projection sur le simplexe (proj_simplex),
          identique a la contrainte utilisee par P3MG/PMMS. Un simple ReLU
          (positivite seule, sans somme=1) donnait a ce modele un espace de
          solutions strictement plus large que les autres baselines, ce qui
          biaisait toute comparaison en sa faveur.
        - Le prox dual correspond au prox exact de la conjuguee de l'attache
          aux donnees L2, F(z) = 0.5*||z - y||^2, dont la conjuguee est
          F*(d) = 0.5*||d||^2 + <d, y>. Le prox exact est
          prox_{sigma F*}(z) = (z - sigma*y) / (1 + sigma). Omettre la
          division par (1 + sigma) transformait cette etape en simple pas
          de gradient (sans regularisation proximale), permettant a la
          variable duale de croitre sans controle et au modele de
          surajuster les observations y.
        """
        Hmat, L2 = sub_static
        p, d = w

        sigma = self.margin / (tau * L2 + 1e-12)

        # 1. Mise a jour du primal : descente + retrecissement (lambda_reg)
        #    + projection sur le simplexe.
        bp = -tau * tc.matmul(d, Hmat)
        p_shrunk = (p + bp) / (1.0 + tau * lambda_reg)
        p_new = proj_simplex(p_shrunk)

        # 2. Extrapolation du primal (over-relaxation)
        p_bar = 2 * p_new - p

        # 3. Mise a jour du dual a partir du primal extrapole (prox exact)
        z = d + sigma * tc.matmul(p_bar, Hmat.t())
        d_new = (z - sigma * y) / (1.0 + sigma)

        w_new = [p_new, d_new]
        return w_new
