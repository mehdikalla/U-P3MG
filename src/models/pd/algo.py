import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat

class PD_Standalone_algo(nn.Module):
    def __init__(self):
        super().__init__()

    def init_PD(self, x0, y):
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        
        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )
        
        # Initialisation du dual (d) et du primal (p)
        p0 = x0
        d0 = tc.zeros_like(tc.matmul(x0, Hmat.t()))
        
        return Hmat, p0, d0

    def iter_PD(self, p, p_old, d, d_old, y, Hmat, tau, sigma, rho):
        """
        Itération Primal-Dual autonome.
        """
        # Mise à jour du primal
        bp = -tau * tc.matmul((2 * d - d_old), Hmat)
        p_new = tc.nn.functional.relu(p + bp) # Activation primal (ex: ReLU)

        # Mise à jour du dual
        bd = sigma * tc.matmul(p_new, Hmat.t())
        
        # Projection/Activation dual (proximal par rapport à y)
        d_temp = d + bd
        d_new = (d_temp - sigma * y) / (1 + sigma * rho) # Simplification L2

        return p_new, d_new