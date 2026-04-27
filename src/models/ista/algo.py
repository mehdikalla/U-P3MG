import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat

def soft_thresholding(x, threshold):
    """
    Opérateur de seuillage doux (Soft Thresholding).
    shrink(x, lambda) = sign(x) * max(|x| - lambda, 0)
    """
    return tc.sign(x) * tc.nn.functional.relu(tc.abs(x) - threshold)

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

    def iter_ISTA(self, x, y, Hmat, gamma, lmbd):
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
        # Le seuil effectif est gamma * lambda
        threshold = gamma * lmbd
        x_new = soft_thresholding(z, threshold)

        return x_new