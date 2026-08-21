import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat

def soft_thresholding(x, threshold):
    """
    Opérateur de seuillage doux (Soft Thresholding).
    shrink(x, lambda) = sign(x) * max(|x| - lambda, 0)
    """
    return tc.sign(x) * tc.nn.functional.relu(tc.abs(x) - threshold)

def proj_simplex_ista(x, eta=1.0):
    """
    Projection euclidienne sur le simplexe {v : sum(v) = eta, v >= 0},
    appliquee ligne par ligne pour un batch (P, N), entierement vectorisee
    (algorithme de tri + cumsum de Held-Wolfe-Crowder / Duchi et al. 2008).

    NOTE: `src.utils.functions.proj_simplex` (utilisee par P3MG/PMMS) boucle
    en Python sur chaque echantillon du batch, ce qui est ~15-20x plus lent
    (mesure empirique) que la version vectorisee ci-dessous. Comme ISTA
    appelle cette projection a CHAQUE couche deroulee, ce cout se multiplie
    par num_layers et dominait le temps d'entrainement. Une projection
    dediee, rapide et verifiee independamment est donc utilisee ici, sans
    modifier le fichier partage `functions.py`.
    """
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)

    P, N = x.shape
    sorted_x, _ = tc.sort(x, dim=1, descending=True)
    cssx = tc.cumsum(sorted_x, dim=1)
    idx = tc.arange(1, N + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
    cond = sorted_x - (cssx - eta) / idx > 0
    # Nombre d'elements verifiant la condition, pour chaque ligne (rho).
    rho = cond.to(x.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    rho_idx = (rho.long() - 1).clamp(min=0)
    css_rho = tc.gather(cssx, 1, rho_idx)
    theta = (css_rho - eta) / rho
    x_proj = tc.clamp(x - theta, min=0.0)

    return x_proj.reshape(orig_shape)

class ISTA_algo(nn.Module):
    def __init__(self):
        super().__init__()

    def init_ISTA(self, x0, y):
        """
        Initialise les variables statiques (Matrice H, constante L).
        """
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        
        # Génération de Hmat (DOSY) via src.utils.functions
        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        # Calcul de la constante de Lipschitz L = lambda_max(H^T H)
        # Pour une matrice réelle, c'est le carré de la norme spectrale (norme 2)
        L = tc.linalg.norm(Hmat, ord=2) ** 2
        
        return Hmat, L

    def iter_ISTA(self, x, y, Hmat, gamma, lmbd, project_simplex=True):
        """
        Une itération de l'algorithme ISTA.
        x_{k+1} = shrink(x_k - gamma * H^T (H x_k - y), gamma * lambda)

        """
        # 1. Calcul du gradient : H^T (H x - y)
        Hx = tc.matmul(x, Hmat.t()) # (Batch, M)
        res = Hx - y
        grad = tc.matmul(res, Hmat) # (Batch, N)

        # 2. Descente de gradient
        z = x - gamma * grad

        # 3. Opérateur proximal (Shrinkage)
        threshold = gamma * lmbd
        x_new = soft_thresholding(z, threshold)

        # 4. Projection sur le simplexe (contrainte physique du probleme DOSY)
        if project_simplex:
            x_new = proj_simplex_ista(x_new)

        return x_new