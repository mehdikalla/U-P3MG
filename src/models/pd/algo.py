import torch as tc
import torch.nn as nn
from src.utils.functions import dosy_mat, proj_simplex


class PD_Standalone_algo(nn.Module):
    """
    Algorithme Primal-Dual autonome (schema de Chambolle-Pock), recode pour
    suivre fidelement la structure de PrimalDual_algo
    (src/models/p3mg/primal_dual/algo.py), et pas seulement son nommage :

    - delta0, gamma0 sont des pas FIXES, deduits une fois pour toutes des
      quantites statiques (ici 1/sqrt(L2), L2 = ||H||^2), exactement comme
      delta0 = gamma0 = 1/norm_D dans PrimalDual_algo. Ils garantissent par
      construction la condition de stabilite delta0*gamma0*||H||^2 <= 1
      (via self.margin).
    - tau n'est PAS un pas de descente mais un coefficient de relaxation
      (sous/sur-relaxation), applique en fin d'iteration exactement comme
      dans PrimalDual_algo.iter_PD :
          un_new = un + tau * (pn - un)
          vn_new = vn + tau * (qn - vn)
      Avec tau = 1, on retrouve l'iteration de Chambolle-Pock standard
      (aucune relaxation). tau est donc, comme dans PrimalDual_algo, fixe
      (non recherche) : il ne modifie pas le probleme resolu a convergence.
    - Le seul hyperparametre desormais recherche par le random search est
      lambda_reg, qui pondere un terme de regularisation quadratique
      reellement present dans le probleme resolu :
          min_x  0.5 * ||H x - y||^2  +  0.5 * lambda_reg * ||x||^2
          s.c.   x dans le simplexe (positivite + somme = 1)
      Contrairement a tau (qui ne fait que regler la vitesse/relaxation de
      convergence vers l'unique minimiseur du probleme non regularise),
      lambda_reg modifie reellement ce probleme : la loss de calibration
      varie donc effectivement avec lambda_reg, y compris a convergence
      complete (grand nombre d'iterations).
    """

    def __init__(self, margin: float = 0.99):
        super().__init__()
        # Marge de securite sous la borne critique de stabilite du schema de
        # Chambolle-Pock (delta0 * gamma0 * ||H||^2 <= 1).
        self.margin = margin

    def init_PD(self, x0, y):
        """
        Initialisation des variables et calculs preliminaires pour le
        primal-dual, alignee sur PrimalDual_algo.init_PD.

        Retourne :
            w_new = [un, vn] : etat initial (primal, dual)
            sub_static = [Hmat, delta0, gamma0] : quantites statiques
                (matrice d'observation et pas fixes delta0/gamma0),
                reutilisees a chaque iteration. delta0 = gamma0 =
                margin / sqrt(L2), avec L2 = ||H||^2 (analogue de
                delta0 = gamma0 = 1/norm_D dans PrimalDual_algo).
        """
        P, N, M = x0.size(0), x0.size(1), y.size(1)

        T, Hmat = dosy_mat(
            int(N), int(M), 0, 1.5, 1, 1000, dtype=x0.dtype, device=x0.device
        )

        # Initialisation du primal (un) et du dual (vn)
        un = x0
        vn = tc.zeros_like(tc.matmul(x0, Hmat.t()))

        # Pas fixes delta0 = gamma0 = margin / ||H|| (sans retropropagation).
        # Garantit delta0 * gamma0 * ||H||^2 <= margin^2 <= 1.
        with tc.no_grad():
            L2 = tc.linalg.matrix_norm(Hmat, ord=2) ** 2
            step0 = self.margin / tc.sqrt(L2 + 1e-12)
        delta0 = step0
        gamma0 = step0

        w_new = [un, vn]
        sub_static = [Hmat, delta0, gamma0]

        return w_new, sub_static

    def iter_PD(self, sub_static, w_new, y, tau, lambda_reg=0.0, q_d=1, q_g=1):
        """
        Iteration generique du Primal-Dual autonome, structuree comme
        PrimalDual_algo.iter_PD.

        Probleme resolu :
            min_x  0.5 * ||H x - y||^2  +  0.5 * lambda_reg * ||x||^2
            s.c.   x dans le simplexe (positivite + somme = 1)

        Etapes (miroir de PrimalDual_algo.iter_PD) :
        1. delta = q_d * delta0, gamma = q_g * gamma0 (pas fixes, eventuel-
           lement moduls par q_d/q_g comme dans PrimalDual_algo).
        2. Calcul du candidat primal pn par prox de l'attache aux donnees
           et de la regularisation quadratique (lambda_reg), suivi de la
           projection sur le simplexe :
               pn = proj_simplex( (un - delta * H^T vn) / (1 + delta * lambda_reg) )
        3. Calcul du candidat dual qn par prox exact de la conjuguee de
           l'attache aux donnees L2, a partir du primal sur-relaxe (2pn-un) :
               v  = vn + gamma * H * (2*pn - un)
               qn = (v - gamma * y) / (1 + gamma)
        4. Relaxation finale (sous/sur-relaxation), exactement comme dans
           PrimalDual_algo :
               un_new = un + tau * (pn - un)
               vn_new = vn + tau * (qn - vn)

        tau est un coefficient de relaxation FIXE (non recherche), transmis
        explicitement a chaque appel. Avec tau = 1, l'iteration se reduit
        exactement au schema de Chambolle-Pock standard (un_new = pn,
        vn_new = qn). tau ne modifie pas le probleme resolu ci-dessus ; a
        convergence, sa valeur (dans la plage de stabilite, tau in (0, 2))
        n'a donc aucun effet sur la loss finale -- d'ou la necessite de
        lambda_reg, seul hyperparametre desormais recherche par le random
        search pour ce modele (cf. src/strategies/random_search.py).

        Contraintes/prox alignes sur les autres modeles de reference
        (P3MG, PMMS, ISTA) :
        - Le prox primal est la projection sur le simplexe (proj_simplex),
          identique a la contrainte utilisee par P3MG/PMMS.
        - Le prox dual correspond au prox exact de la conjuguee de l'attache
          aux donnees L2, F(z) = 0.5*||z - y||^2, dont la conjuguee est
          F*(v) = 0.5*||v||^2 + <v, y>. Le prox exact est
          prox_{gamma F*}(z) = (z - gamma*y) / (1 + gamma).
        """
        Hmat, delta0, gamma0 = sub_static
        un, vn = w_new

        delta = q_d * delta0
        gamma = q_g * gamma0

        # 2. Candidat primal : descente + retrecissement (lambda_reg) +
        #    projection sur le simplexe.
        bp = -delta * tc.matmul(vn, Hmat)
        u_shrunk = (un + bp) / (1.0 + delta * lambda_reg)
        pn = proj_simplex(u_shrunk)

        # 3. Candidat dual : prox exact, a partir du primal sur-relaxe.
        v = vn + gamma * tc.matmul(2 * pn - un, Hmat.t())
        qn = (v - gamma * y) / (1.0 + gamma)

        # 4. Relaxation finale (tau = 1 -> schema standard, sans relaxation).
        un_new = un + tau * (pn - un)
        vn_new = vn + tau * (qn - vn)

        w_new = [un_new, vn_new]
        return w_new
