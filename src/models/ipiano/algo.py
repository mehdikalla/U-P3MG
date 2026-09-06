"""
Algorithme iPiano (Inertial Proximal Algorithm for Non-convex Optimization)
pour la deconvolution parcimonieuse avec penalite log(l1/l2) et contrainte
de simplexe.

Reecriture directe (batchee, PyTorch) de la reference MATLAB
`runiPIANOalgorithm.m` :

    G(x) = F(x) + R(x)
    F(x) = 1/2||Hx - y||^2 + nu * log( (l1(x) + beta) / l2(x) )
    R(x) = iota_{simplexe}(x)

A chaque iteration, un pas de descente inertiel (forward) est effectue :

    temp = x - alpha * grad F(x) + beta_ipiano * (x - x_prec)
    x_new = Proj_simplexe(temp)

avec un pas `alpha = 1/99 * (1 - beta_ipiano) / LipsBack` recherche par
"backtracking" (recherche lineaire retrograde) sur la constante de
Lipschitz locale `LipsBack`, de sorte que la condition de descente
suivante soit satisfaite :

    F(x_new) <= F(x) + grad F(x)^T (x_new - x) + LipsBack/2 * ||x_new - x||^2

Sinon, `LipsBack` est multiplie par 4 et le pas est recalcule (jusqu'a
`backtrials` tentatives).

Seul l'hyperparametre `nu` est destine a etre calibre (cf.
`src/strategies/random_search.py`, qui explore `nu` par recherche
log-uniforme, exactement comme pour PMMS) ; `sigma`, `beta`, `eta` sont des
parametres physiques fixes (cf. config.yaml).
"""

import torch as tc
import torch.nn as nn

from src.utils.functions import Criterion, gradient_x, proj_simplex

# Constante d'inertie iPiano (poids du terme momentum (x - x_prec)),
# reprise telle quelle de `runiPIANOalgorithm.m`.
BETA_IPIANO = 0.8
# Nombre maximal de tentatives de recherche lineaire retrograde par iteration.
BACKTRIALS = 20
# Valeur initiale de la constante de Lipschitz locale (mise a jour de
# maniere persistante entre iterations, comme dans la reference MATLAB).
LIPS_BACK_INIT = 1.0


class IPIANO_algo(nn.Module):
    """Encapsule l'initialisation et une iteration de l'algorithme iPiano.

    Les tenseurs sont batches sur la premiere dimension (P = taille de
    batch), conformement aux autres algorithmes du depot (P3MG, PMMS, ISTA,
    HQ).
    """

    def __init__(self):
        super().__init__()

    def init_IPIANO(
        self,
        x0: tc.Tensor,
        y: tc.Tensor,
        sigma: float = 1e-5,
        beta: float = 1e-5,
        eta: float = 1e-2,
    ):
        """Initialise les variables statiques et dynamiques de iPiano.

        Args:
            x0: signal initial, shape (P, N).
            y: observations, shape (P, M).
            sigma: parametre de lissage de la norme l1 (SOOT), doit etre
                egal a alpha dans la convention du depot.
            beta: parametre de stabilisation du denominateur log(l1+beta).
            eta: parametre de lissage de la norme l2.

        Returns:
            static: (Hmat, sigma, beta, eta)
            dynamic: (x_old, LipsBack)
        """
        from src.utils.functions import dosy_mat

        P, N, M = x0.size(0), x0.size(1), y.size(1)

        _, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        x_old = x0.clone()
        LipsBack = tc.full((P, 1), LIPS_BACK_INIT, dtype=x0.dtype, device=x0.device)

        static = (Hmat, sigma, beta, eta)
        dynamic = (x_old, LipsBack)

        return static, dynamic

    def iter_IPIANO(self, static, dynamic, x: tc.Tensor, y: tc.Tensor, nu):
        """Effectue une iteration de l'algorithme iPiano.

        Reproduit fidelement la boucle `for iter = 1:NbIt` de
        `runiPIANOalgorithm.m` :
            1. gradient de F en x ;
            2. recherche lineaire retrograde sur LipsBack : pas inertiel
               `alpha = 1/99*(1-beta_ipiano)/LipsBack`, projection sur le
               simplexe, jusqu'a satisfaire la condition de descente
               majorante (ou epuisement de `BACKTRIALS`) ;
            3. mise a jour x_prec <- x, x <- x_new.

        Args:
            static: sortie de `init_IPIANO`.
            dynamic: sortie de `init_IPIANO` ou de l'iteration precedente.
            x: iterate courant, shape (P, N).
            y: observations, shape (P, M).
            nu: poids de la penalite log(l1/l2), scalaire ou tenseur.

        Returns:
            x_new: iterate mis a jour, shape (P, N).
            dynamic_new: nouvel etat dynamique.
        """
        Hmat, sigma, beta, eta = static
        x_old, LipsBack = dynamic

        grad_x, _ = gradient_x(x, y, Hmat, sigma, beta, eta, nu)
        Crit = Criterion(x, Hmat, y, sigma, beta, eta, nu)

        P = x.size(0)
        x_new = x.clone()
        LipsBack_new = LipsBack.clone()
        # Masque des elements de batch dont la recherche lineaire n'a pas
        # encore satisfait la condition de descente majorante.
        active = tc.ones((P, 1), dtype=tc.bool, device=x.device)

        for _ in range(BACKTRIALS):
            if not tc.any(active):
                break

            alpha_ipiano = (1.0 / 99.0) * (1.0 - BETA_IPIANO) / LipsBack_new
            temp = x - alpha_ipiano * grad_x + BETA_IPIANO * (x - x_old)
            x_candidate = proj_simplex(temp)

            crit_back = Criterion(x_candidate, Hmat, y, sigma, beta, eta, nu)
            diff = x_candidate - x
            majorant = (
                Crit
                + tc.sum(grad_x * diff, dim=1, keepdim=True)
                + LipsBack_new / 2 * tc.sum(diff * diff, dim=1, keepdim=True)
            )
            fails = (crit_back > majorant) & active

            # Les elements qui satisfont la condition sont figes (retires
            # du masque actif) ; leur x_candidate est conserve.
            x_new = tc.where(active & ~fails, x_candidate, x_new)
            LipsBack_new = tc.where(fails, LipsBack_new * 4.0, LipsBack_new)
            active = fails

        # Pour les elements de batch n'ayant jamais satisfait la condition
        # apres BACKTRIALS tentatives, on conserve le dernier candidat
        # calcule (comportement de la boucle `for` MATLAB, qui sort sans
        # `break` explicite en cas d'echec final).
        if tc.any(active):
            alpha_ipiano = (1.0 / 99.0) * (1.0 - BETA_IPIANO) / LipsBack_new
            temp = x - alpha_ipiano * grad_x + BETA_IPIANO * (x - x_old)
            x_candidate = proj_simplex(temp)
            x_new = tc.where(active, x_candidate, x_new)

        dynamic_new = (x, LipsBack_new)

        return x_new, dynamic_new
