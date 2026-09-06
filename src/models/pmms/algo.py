"""
Algorithme PMMS (Proximal Majorize-Minimize Subspace) pour la deconvolution
parcimonieuse avec penalite log(l1/l2) et contrainte de simplexe.

Reecriture directe (batchee, PyTorch) de la reference MATLAB
`runPMMSalgorithm.m` :

    G(x) = F(x) + R(x)
    F(x) = 1/2||Hx - y||^2 + nu * log( (l1(x) + beta) / l2(x) )
    R(x) = iota_{simplexe}(x)   approchee par une penalite quadratique
           de poids croissant `gamma` (methode de penalisation exterieure),
           avec mise a jour de gamma/epsilon selon le schema :
               gamma_j   = (3*j)^{3/2}
               epsilon_j = 10 / gamma_j^{1/4}
           incremente (j <- j+1) des que ||grad F(x)|| <= epsilon_j.

A chaque iteration, l'algorithme construit un sous-espace de descente
D = [-grad, dx_prec] (memoire du pas precedent, cf. 3MG "sous-espace de
memoire de gradient"), majore la fonction dans ce sous-espace via une
matrice B = D^T A D (A = Hessienne majorante de F, diagonale + H^T H),
puis resout le probleme de minimisation quadratique reduit par
pseudo-inverse.

Seul l'hyperparametre `nu` est destine a etre calibre (cf.
`src/strategies/random_search.py`, qui explore `nu` par recherche
log-uniforme) ; `sigma`, `beta`, `eta` sont des parametres physiques fixes
(cf. config.yaml).
"""

from typing import Tuple

import torch as tc
import torch.nn as nn

from src.utils.functions import (
    LipschitzSOOT,
    Majorante_x,
    gradient_x,
    proj_simplex,
)

# Constantes du schema de mise a jour de la penalisation exterieure
# (gamma, epsilon), reprises telles quelles de `runPMMSalgorithm.m`.
_CTE_GAMMA = 3.0
_EXP_GAMMA = 1.5
_CTE_EPSILON = 10.0
_EXP_EPSILON = 0.25


def _gamma_epsilon(countj: tc.Tensor) -> Tuple[tc.Tensor, tc.Tensor]:
    """Calcule (gamma, epsilon) a partir du compteur d'iterations `countj`.

    Reproduit :
        gamma   = (cte_gamma * countj) ** exp_gamma
        epsilon = cte_epsilon / gamma ** exp_epsilon
    """
    gamma = (_CTE_GAMMA * countj) ** _EXP_GAMMA
    epsilon = _CTE_EPSILON / (gamma ** _EXP_EPSILON)
    return gamma, epsilon


class PMMS_algo(nn.Module):
    """Encapsule l'initialisation et une iteration de l'algorithme PMMS.

    Les tenseurs sont batches sur la premiere dimension (P = taille de
    batch), conformement aux autres algorithmes du depot (P3MG, ISTA, HQ).
    """

    def __init__(self):
        super().__init__()

    def init_PMMS(
        self,
        x0: tc.Tensor,
        y: tc.Tensor,
        sigma: float = 1e-5,
        beta: float = 1e-5,
        eta: float = 1e-2,
    ):
        """Initialise les variables statiques et dynamiques de PMMS.

        Args:
            x0: signal initial, shape (P, N).
            y: observations, shape (P, M).
            sigma: parametre de lissage de la norme l1 (SOOT), doit etre
                egal a alpha dans la convention du depot.
            beta: parametre de stabilisation du denominateur log(l1+beta).
            eta: parametre de lissage de la norme l2.

        Returns:
            static: (Hmat, sigma, beta, eta, Cg2, Lips)
            dynamic: (dx_old, countj, gamma_pen, epsilon, iter_count)
        """
        from src.utils.functions import dosy_mat

        P, N, M = x0.size(0), x0.size(1), y.size(1)

        _, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        # Constante de majoration du terme quadratique de la norme l2
        # (cf. Cg2 = 9/(N*8*eta^2) dans runPMMSalgorithm.m).
        Cg2 = 9.0 / (N * 8 * eta ** 2)

        # Constante de Lipschitz du terme SOOT (diagnostic uniquement ;
        # n'intervient pas dans la mise a jour de x elle-meme).
        Lips = LipschitzSOOT(sigma, beta, eta, N)

        countj = tc.ones((P, 1), dtype=x0.dtype, device=x0.device)
        gamma_pen, epsilon = _gamma_epsilon(countj)

        dx_old = tc.zeros_like(x0)
        iter_count = 1

        static = (Hmat, sigma, beta, eta, Cg2, Lips)
        dynamic = (dx_old, countj, gamma_pen, epsilon, iter_count)

        return static, dynamic

    def iter_PMMS(self, static, dynamic, x: tc.Tensor, y: tc.Tensor, nu):
        """Effectue une iteration de l'algorithme PMMS.

        Reproduit fidelement la boucle `for iter = 1:NbIt` de
        `runPMMSalgorithm.m` :
            1. gradient de G (F + penalite de simplexe) en x ;
            2. sous-espace D = [-grad] (iteration 1) ou [-grad, dx_prec] ;
            3. majorante A du terme F dans ce sous-espace (matrice B) ;
            4. minimisation quadratique reduite (pseudo-inverse) -> dx ;
            5. mise a jour x <- x + dx ;
            6. mise a jour (gamma, epsilon, countj) si ||grad|| <= epsilon.

        Args:
            static: sortie de `init_PMMS`.
            dynamic: sortie de `init_PMMS` ou de l'iteration precedente.
            x: iterate courant, shape (P, N).
            y: observations, shape (P, M).
            nu: poids de la penalite log(l1/l2), scalaire ou tenseur.

        Returns:
            x_new: iterate mis a jour, shape (P, N).
            dynamic_new: nouvel etat dynamique.
        """
        Hmat, sigma, beta, eta, Cg2, _Lips = static
        dx_old, countj, gamma_pen, epsilon, iter_count = dynamic

        # 1. Gradient de F (attache aux donnees + log(l1/l2)).
        grad_base, l1 = gradient_x(x, y, Hmat, sigma, beta, eta, nu)

        # Gradient de la penalite de simplexe : gamma * (x - Proj_simplexe(x)).
        Px = proj_simplex(x)
        gradx = grad_base + gamma_pen * (x - Px)

        # 2. Sous-espace de descente D = [-grad] ou [-grad, dx_precedent].
        Dx_list = [-gradx]
        if iter_count > 1:
            Dx_list.append(dx_old)
        D = tc.stack(Dx_list, dim=1)  # (P, L, N)

        # 3. Majorante de F dans le sous-espace + contribution de la
        # penalite de simplexe (matrice gamma * Identite).
        Ad_list = []
        for d_vec in Dx_list:
            Ad_base = Majorante_x(d_vec, x, sigma, l1, beta, Cg2, Hmat, nu)
            Ad_list.append(Ad_base + gamma_pen * d_vec)
        Ad = tc.stack(Ad_list, dim=1)  # (P, L, N)

        B = tc.bmm(D, Ad.transpose(1, 2))  # (P, L, L)
        # Symetrisation : stabilite numerique du pseudo-inverse (B est
        # theoriquement symetrique, mais les erreurs d'arrondi peuvent
        # introduire une legere asymetrie).
        B = 0.5 * (B + B.transpose(1, 2))

        # 4. Minimisation quadratique reduite : u = -pinv(B) * D * grad,
        # dx = D^T * u.
        D_gradx = tc.bmm(D, gradx.unsqueeze(2))
        B_pinv = tc.linalg.pinv(B)
        u = -tc.bmm(B_pinv, D_gradx)
        dx_new = tc.bmm(u.transpose(1, 2), D).squeeze(1)

        x_new = x + dx_new

        # 5. Mise a jour du schema de penalisation exterieure
        # (gamma, epsilon) des que la norme du gradient de F passe sous
        # le seuil epsilon courant.
        grad_norm = tc.norm(gradx, dim=1, keepdim=True)
        update_mask = (grad_norm <= epsilon).type(x.dtype)

        countj_new = countj + update_mask
        gamma_new, epsilon_new_candidate = _gamma_epsilon(countj_new)
        gamma_pen_new = tc.where(update_mask > 0, gamma_new, gamma_pen)
        epsilon_new = tc.where(update_mask > 0, epsilon_new_candidate, epsilon)

        dynamic_new = (dx_new, countj_new, gamma_pen_new, epsilon_new, iter_count + 1)

        return x_new, dynamic_new