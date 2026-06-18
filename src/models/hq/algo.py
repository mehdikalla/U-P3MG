import torch as tc
import torch.nn as nn

# Importation centralisée depuis les utilitaires
from src.utils.functions import (
    dosy_mat,
    phi_prime_cvx,
    weight_cvx,
    phi_prime_ncvx,
    weight_ncvx
)

class HQ_algo(nn.Module):
    def __init__(self):
        super().__init__()

    def init_HQ(self, x0, y):
        """
        Initialise les variables statiques pour le modèle HQ.
        Précalcule la matrice H^T H pour optimiser l'inversion MM.
        """
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        
        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )
        
        Ht_y = tc.matmul(y, Hmat)
        Ht_H = tc.matmul(Hmat.t(), Hmat)
        
        return Hmat, Ht_y, Ht_H

    def iter_HQ(self, x, y, Hmat, Ht_y, Ht_H, gamma, lmbd_cvx, lmbd_ncvx, delta_cvx=0.01, delta_ncvx=0.01):
        """
        Exécute une itération de la Minimisation Majorée (MM) pour Half-Quadratic.
        """
        P, N = x.shape
        
        # 1. Attache aux données : H^T (H x) - H^T y
        Hx = tc.matmul(x, Hmat.t())
        first_branch = tc.matmul(Hx, Hmat) - Ht_y

        # 2. Gradients de pénalisation
        second_branch_cvx = phi_prime_cvx(x, delta_cvx)
        second_branch_ncvx = phi_prime_ncvx(x, delta_ncvx)
        
        if isinstance(lmbd_cvx, tc.Tensor) and lmbd_cvx.dim() == 1:
            lmbd_cvx = lmbd_cvx.unsqueeze(1)
            lmbd_ncvx = lmbd_ncvx.unsqueeze(1)
            
        grad_total = first_branch + lmbd_cvx * second_branch_cvx + lmbd_ncvx * second_branch_ncvx

        # 3. Poids de la matrice majorante
        w_cvx = weight_cvx(x, delta_cvx)
        w_ncvx = weight_ncvx(x, delta_ncvx)
        
        # ---------------------------------------------------------------------
        # OPTIMISATION : Retour à `solve`
        # L'ajout de eps garantit que la matrice est définie positive.
        # `solve` est nativement optimisé pour les GPU (résolution LU) et 
        # son backward est infiniment plus rapide que celui de `lstsq`.
        # ---------------------------------------------------------------------
        eps = 1e-5 
        D_diag = lmbd_cvx * w_cvx + lmbd_ncvx * w_ncvx + eps
        
        # Construction du système A = H^T H + diag(D)
        Ht_H_batch = Ht_H.unsqueeze(0).expand(P, -1, -1)
        D_matrix = tc.diag_embed(D_diag)
        A_mat = Ht_H_batch + D_matrix
        
        # Résolution linéaire standard
        grad_total_unsq = grad_total.unsqueeze(2)
        direction = tc.linalg.solve(A_mat, grad_total_unsq).squeeze(2)

        # 4. Descente et projection sur le quadrant positif (ReLU)
        x_new = x - gamma * direction

        return tc.nn.functional.relu(x_new)