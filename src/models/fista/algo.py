"""
Algorithme FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) pour la
deconvolution parcimonieuse avec penalite log(l1/l2) et contrainte de
simplexe.

Reecriture directe (batchee, PyTorch) de la reference MATLAB
`runFISTAalgorithm.m` :

    G(x) = F(x) + R(x)
    F(x) = 1/2||Hx - y||^2 + nu * log( (l1(x) + beta) / l2(x) )
    R(x) = iota_{simplexe}(x)

A chaque iteration, un pas forward-backward accelere (extrapolation de
Nesterov) est effectue, avec un pas de descente global constant
`1/Lips` (Lips = constante de Lipschitz globale du gradient de F,
majorant somme de la partie SOOT lissee et de ||H||^2) :

    forward   = x - grad F(x) / Lips
    x_fista   = Proj_simplexe(forward)
    t_new     = (1 + sqrt(1 + 4*t^2)) / 2
    x_new     = x_fista + (t-1)/t_new * (x_fista - x_fista_prec)

Seul l'hyperparametre `nu` est destine a etre calibre (cf.
`src/strategies/random_search.py`, qui explore `nu` par recherche
log-uniforme, exactement comme pour PMMS, iPiano et VMFB) ; `sigma`,
`beta`, `eta` sont des parametres physiques fixes (cf. config.yaml).
"""

import torch as tc
import torch.nn as nn

from src.utils.functions import LipschitzSOOT, gradient_x, proj_simplex


class FISTA_algo(nn.Module):
    """Encapsule l'initialisation et une iteration de l'algorithme FISTA.

    Les tenseurs sont batches sur la premiere dimension (P = taille de
    batch), conformement aux autres algorithmes du depot (P3MG, PMMS,
    iPiano, VMFB, ISTA, HQ).
    """

    def __init__(self):
        super().__init__()

    def init_FISTA(
        self,
        x0: tc.Tensor,
        y: tc.Tensor,
        sigma: float = 1e-5,
        beta: float = 1e-5,
        eta: float = 1e-2,
    ):
        """Initialise les variables statiques et dynamiques de FISTA.

        Args:
            x0: signal initial, shape (P, N).
            y: observations, shape (P, M).
            sigma: parametre de lissage de la norme l1 (SOOT), doit etre
                egal a alpha dans la convention du depot.
            beta: parametre de stabilisation du denominateur log(l1+beta).
            eta: parametre de lissage de la norme l2.

        Returns:
            static: (Hmat, sigma, beta, eta, Lips)
            dynamic: (x_fista_old, t)
        """
        from src.utils.functions import dosy_mat

        P = x0.size(0)
        N = x0.size(1)
        M = y.size(1)

        _, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        # Constante de Lipschitz globale du gradient de F : partie
        # quadratique (||H||^2). La partie SOOT (nu * LipschitzSOOT) depend
        # de `nu`, calibrable et connu seulement au moment de l'iteration :
        # elle est donc recalculee dynamiquement dans `iter_FISTA`, fidele a
        # `Lips = nu*LipschitzSOOT(...) + Hnorm2` dans `runFISTAalgorithm.m`.
        Hnorm2 = tc.linalg.matrix_norm(Hmat, ord=2) ** 2

        x_fista_old = x0.clone()
        t = tc.ones((P, 1), dtype=x0.dtype, device=x0.device)

        static = (Hmat, sigma, beta, eta, Hnorm2)
        dynamic = (x_fista_old, t)

        return static, dynamic

    def iter_FISTA(self, static, dynamic, x: tc.Tensor, y: tc.Tensor, nu):
        """Effectue une iteration de l'algorithme FISTA.

        Reproduit fidelement la boucle `for iter = 1:NbIt` de
        `runFISTAalgorithm.m` :
            1. gradient de F en x ;
            2. pas forward a pas constant 1/Lips + projection simplexe ;
            3. extrapolation de Nesterov (mise a jour de t et x).

        Args:
            static: sortie de `init_FISTA`.
            dynamic: sortie de `init_FISTA` ou de l'iteration precedente.
            x: iterate courant (point extrapole), shape (P, N).
            y: observations, shape (P, M).
            nu: poids de la penalite log(l1/l2), scalaire ou tenseur.

        Returns:
            x_new: iterate mis a jour (point extrapole suivant), shape (P, N).
            dynamic_new: nouvel etat dynamique.
        """
        Hmat, sigma, beta, eta, Hnorm2 = static
        x_fista_old, t = dynamic

        N = x.size(1)
        Lipsoot = nu * LipschitzSOOT(sigma, beta, eta, N)
        Lips = Lipsoot + Hnorm2

        grad_x, _ = gradient_x(x, y, Hmat, sigma, beta, eta, nu)

        forward = x - grad_x / Lips
        x_fista = proj_simplex(forward)

        t_new = (1.0 + tc.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        x_new = x_fista + ((t - 1.0) / t_new) * (x_fista - x_fista_old)

        dynamic_new = (x_fista, t_new)

        return x_new, dynamic_new
