"""
Algorithme VMFB (Variable Metric Forward-Backward) pour la deconvolution
parcimonieuse avec penalite log(l1/l2) et contrainte de simplexe.

Reecriture directe (batchee, PyTorch) de la reference MATLAB
`runVMFBalgorithm.m` :

    G(x) = F(x) + R(x)
    F(x) = 1/2||Hx - y||^2 + nu * log( (l1(x) + beta) / l2(x) )
    R(x) = iota_{simplexe}(x)

A chaque iteration, un pas forward-backward a metrique variable (diagonale)
est effectue :

    A       = majorante diagonale de F en x (cf. `Majorante_x` de
              `src/utils/functions.py`, deja utilisee par P3MG/PMMS)
    forward = x - gamma * grad F(x) / A
    x_new   = Proj_simplexe^{gamma/A}(forward)

ou `Proj_simplexe^{w}` designe la projection au sens de la metrique
diagonale de poids `w` (cf. `proj_weighted_simplex` dans
`src/utils/functions.py`), et `gamma = 1.9` est le pas de relaxation fixe
de la reference MATLAB (converge tant que gamma < 2, la majorante
diagonale `A` etant deja une borne de Lipschitz locale du gradient).

Seul l'hyperparametre `nu` est destine a etre calibre (cf.
`src/strategies/random_search.py`, qui explore `nu` par recherche
log-uniforme, exactement comme pour PMMS et iPiano) ; `sigma`, `beta`,
`eta` sont des parametres physiques fixes (cf. config.yaml).
"""

import torch as tc
import torch.nn as nn

from src.utils.functions import gradient_x, majorante_x_diag, proj_weighted_simplex

# Pas de relaxation fixe du schema forward-backward, repris tel quel de
# `runVMFBalgorithm.m` (converge pour 0 < gamma < 2).
GAMMA_VMFB = 1.9


class VMFB_algo(nn.Module):
    """Encapsule l'initialisation et une iteration de l'algorithme VMFB.

    Les tenseurs sont batches sur la premiere dimension (P = taille de
    batch), conformement aux autres algorithmes du depot (P3MG, PMMS,
    iPiano, ISTA, HQ).
    """

    def __init__(self):
        super().__init__()

    def init_VMFB(
        self,
        x0: tc.Tensor,
        y: tc.Tensor,
        sigma: float = 1e-5,
        beta: float = 1e-5,
        eta: float = 1e-2,
    ):
        """Initialise les variables statiques de VMFB.

        Args:
            x0: signal initial, shape (P, N).
            y: observations, shape (P, M).
            sigma: parametre de lissage de la norme l1 (SOOT), doit etre
                egal a alpha dans la convention du depot.
            beta: parametre de stabilisation du denominateur log(l1+beta).
            eta: parametre de lissage de la norme l2.

        Returns:
            static: (Hmat, sigma, beta, eta, Cg2)
            dynamic: () (VMFB ne porte aucun etat entre iterations, hormis
                l'iterate x lui-meme).
        """
        from src.utils.functions import dosy_mat

        N = x0.size(1)
        M = y.size(1)

        _, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        # Constante de majoration du terme quadratique de la norme l2
        # (cf. Cg2 = 9/(N*8*eta^2) dans runVMFBalgorithm.m).
        Cg2 = 9.0 / (N * 8 * eta ** 2)

        # Constante de Lipschitz globale du terme d'attache aux donnees
        # (Hnorm2 = ||H||^2), utilisee comme majorante diagonale de F
        # (cf. Majorante_x/majorante_x_diag, deja partagee avec P3MG).
        Hnorm2 = tc.linalg.matrix_norm(Hmat, ord=2) ** 2

        static = (Hmat, sigma, beta, eta, Cg2, Hnorm2)
        dynamic = ()

        return static, dynamic

    def iter_VMFB(self, static, dynamic, x: tc.Tensor, y: tc.Tensor, nu):
        """Effectue une iteration de l'algorithme VMFB.

        Reproduit fidelement la boucle `for iter = 1:NbIt` de
        `runVMFBalgorithm.m` :
            1. gradient de F en x ;
            2. majorante diagonale A de F en x ;
            3. pas forward : forward = x - gamma * grad / A ;
            4. projection metrique sur le simplexe (poids gamma/A) ;
            5. x <- x_new.

        Args:
            static: sortie de `init_VMFB`.
            dynamic: sortie de `init_VMFB` ou de l'iteration precedente
                (inutilisee, VMFB etant sans memoire, mais conservee par
                coherence d'interface avec les autres algorithmes du
                depot).
            x: iterate courant, shape (P, N).
            y: observations, shape (P, M).
            nu: poids de la penalite log(l1/l2), scalaire ou tenseur.

        Returns:
            x_new: iterate mis a jour, shape (P, N).
            dynamic_new: etat dynamique inchange (tuple vide).
        """
        Hmat, sigma, beta, eta, Cg2, Hnorm2 = static

        grad_x, l1 = gradient_x(x, y, Hmat, sigma, beta, eta, nu)
        A = majorante_x_diag(x, sigma, l1, beta, Cg2, nu, Hnorm2)

        forward = x - GAMMA_VMFB * grad_x / A
        weight = A / GAMMA_VMFB
        x_new = proj_weighted_simplex(forward, weight)

        return x_new, dynamic
