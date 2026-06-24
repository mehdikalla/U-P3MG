import torch as tc
import torch.nn as nn

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
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        T, Hmat = dosy_mat(int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device)
        
        # Ht_y est supprimé d'ici car il dépend de la donnée y qui change à chaque batch
        Ht_H = tc.matmul(Hmat.t(), Hmat).contiguous()
        
        return Hmat, Ht_H

    def iter_HQ(self, x, y, Hmat, Ht_H, gamma, lmbd_cvx, lmbd_ncvx, delta_cvx=0.01, delta_ncvx=0.01):
        P, N = x.shape
        M = y.shape[1]
        
        # Calcul dynamique de Ht_y pour le batch en cours
        Ht_y = tc.matmul(y, Hmat).contiguous()
        
        Hx = tc.matmul(x, Hmat.t().contiguous())
        first_branch = tc.matmul(Hx, Hmat) - Ht_y

        second_branch_cvx = phi_prime_cvx(x, delta_cvx)
        second_branch_ncvx = phi_prime_ncvx(x, delta_ncvx)
        
        if isinstance(lmbd_cvx, tc.Tensor) and lmbd_cvx.dim() == 1:
            lmbd_cvx = lmbd_cvx.unsqueeze(1)
            lmbd_ncvx = lmbd_ncvx.unsqueeze(1)
            
        grad_total = first_branch + lmbd_cvx * second_branch_cvx + lmbd_ncvx * second_branch_ncvx

        w_cvx = weight_cvx(x, delta_cvx)
        w_ncvx = weight_ncvx(x, delta_ncvx)
        
        eps = 1e-2 
        D_diag = lmbd_cvx * w_cvx + lmbd_ncvx * w_ncvx + eps
        
        D_inv = 1.0 / D_diag 
        D_inv_g = D_inv * grad_total 
        
        H_D_inv = Hmat.unsqueeze(0) * D_inv.unsqueeze(1) 
        H_D_inv_Ht = tc.matmul(H_D_inv, Hmat.t()) 
        
        I_M = tc.eye(M, dtype=Hmat.dtype, device=Hmat.device).unsqueeze(0)
        Inner_mat = I_M + H_D_inv_Ht 
        
        Inner_inv = tc.linalg.inv(Inner_mat)
        
        H_D_inv_g = tc.matmul(Hmat, D_inv_g.unsqueeze(2)) 
        temp = tc.matmul(Inner_inv, H_D_inv_g) 
        Ht_temp = tc.matmul(Hmat.t(), temp).squeeze(2) 
        
        correction = D_inv * Ht_temp 
        direction = D_inv_g - correction

        x_new = x - gamma * direction

        return tc.nn.functional.relu(x_new)