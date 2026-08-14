import torch as tc
import torch.nn as nn
from src.utils.functions import *

class PMMS_algo(nn.Module):
    def __init__(self):
        super().__init__()

    def init_PMMS(self, x0, y, sigma=1e-5, beta=1e-5, eta=1e-2):
        """
        Initialisation des variables statiques et dynamiques.
        """
        P, N, M = x0.size(0), x0.size(1), y.size(1)
        
        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        Cg2 = 9.0 / (N * 8 * eta**2)
        
        countj = tc.ones((P, 1), dtype=x0.dtype, device=x0.device)
        gamma_pen = (3.0 * countj) ** 1.5
        epsilon = 10.0 / (gamma_pen ** 0.25)
        
        dx_old = tc.zeros_like(x0)
        iter_count = 1
        
        static = (Hmat, sigma, beta, eta, Cg2)
        dynamic = (dx_old, countj, gamma_pen, epsilon, iter_count)
        
        return static, dynamic

    def iter_PMMS(self, static, dynamic, x, y, nu):
        """
        Itération principale PMMS. L'hyperparamètre nu est passé dynamiquement.
        """
        Hmat, sigma, beta, eta, Cg2 = static
        dx_old, countj, gamma_pen, epsilon, iter_count = dynamic
        
        # 1. Utilisation du gradient existant avec nu optimisé
        grad_base, l1 = gradient_x(x, y, Hmat, sigma, beta, eta, nu)
        
        # 2. Ajout manuel de la pénalité du simplexe spécifique à PMMS
        Px = proj_simplex(x)
        gradx = grad_base + gamma_pen * (x - Px)
        
        # Construction du sous-espace D
        Dx_list = [-gradx]
        if iter_count > 1:
            Dx_list.append(dx_old)
            
        D = tc.stack(Dx_list, dim=1) # (P, L, N)
        
        # Construction de la matrice B
        Ad_list = []
        for d_vec in Dx_list:
            Ad_base = Majorante_x(d_vec, x, sigma, l1, beta, Cg2, Hmat, nu)
            Ad_temp = Ad_base + gamma_pen * d_vec
            Ad_list.append(Ad_temp)
            
        Ad = tc.stack(Ad_list, dim=1) # (P, L, N)
        B = tc.bmm(D, Ad.transpose(1, 2)) # (P, L, L)
        B = 0.5 * (B + B.transpose(1, 2))  # Symetrisation pour la stabilite numerique du pinv

        # Minimisation dans le sous-espace
        D_gradx = tc.bmm(D, gradx.unsqueeze(2))
        B_pinv = tc.linalg.pinv(B)
        u = -tc.bmm(B_pinv, D_gradx)
        
        dx_new = tc.bmm(u.transpose(1, 2), D).squeeze(1)
        x_new = x + dx_new
        
        # Mises à jour dynamiques PMMS
        grad_norm = tc.norm(gradx, dim=1, keepdim=True)
        update_mask = (grad_norm <= epsilon).type(x.dtype)
        
        countj_new = countj + update_mask
        gamma_pen_new = tc.where(update_mask > 0, (3.0 * countj_new) ** 1.5, gamma_pen)
        epsilon_new = tc.where(update_mask > 0, 10.0 / (gamma_pen_new ** 0.25), epsilon)
        
        dynamic_new = (dx_new, countj_new, gamma_pen_new, epsilon_new, iter_count + 1)
        
        return x_new, dynamic_new