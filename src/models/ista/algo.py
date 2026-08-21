import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat, proj_simplex

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

    def iter_ISTA(self, x, y, Hmat, gamma, lmbd, project_simplex=True):
        """
        Une itération de l'algorithme ISTA.
        x_{k+1} = shrink(x_k - gamma * H^T (H x_k - y), gamma * lambda)

        `project_simplex` applique en plus une projection sur le simplexe
        (positivite + somme unitaire) apres le seuillage doux. Ce jeu de
        donnees DOSY genere des signaux `xtrue` verifiant par construction
        `sum(xtrue) == 1` et `xtrue >= 0` (cf. Dataset/data_*/train.pt) : sans
        cette contrainte physique, le simple seuillage doux ISTA ne peut pas
        exploiter cette information et plafonne tres pres d'un baseline
        trivial (zero/moyenne), contrairement a P3MG/PMMS qui projettent
        explicitement sur le simplexe (cf. src/models/p3mg/algo.py,
        src/models/pmms/algo.py, fonction proj_simplex).
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

        # 4. Projection sur le simplexe (contrainte physique du probleme DOSY)
        if project_simplex:
            x_new = proj_simplex(x_new)

        return x_new