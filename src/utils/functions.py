from typing import Union, Optional
import torch as tc
import torch.nn as nn

class Max:
    r"""Compute the proximity operator and the evaluation of gamma*f.
    f(x) = max( x1, ..., xn)
    """

    def __init__(self, axis: Optional[int] = None):
        if (axis is not None) and (axis < 0):
            axis = None
        self.axis = axis

    def prox(self, x: tc.Tensor, gamma: Union[float, tc.Tensor] = 1.0) -> tc.Tensor:
        if isinstance(gamma, tc.Tensor):
            scale = gamma.to(dtype=x.dtype, device=x.device)
        else:
            scale = tc.tensor(gamma, dtype=x.dtype, device=x.device)

        self._check(x, scale)
        axis = self.axis
        sz = x.shape

        if x.numel() <= 1:
            x = x.reshape(-1)

        sz0 = list(x.shape)
        if axis is not None:
            sz0[axis] = 1
        if scale.numel() > 1:
            scale = scale.reshape(sz0)

        if axis is None:
            x = x.reshape(-1)
            scale = scale.reshape(-1)
            sort_x = -tc.sort(-x).values
            ones_ = tc.ones_like(sort_x)
            cum_sum = (tc.cumsum(sort_x, dim=0) - scale) / tc.cumsum(ones_, dim=0)

        else:
            sort_x = -tc.sort(-x, dim=axis).values
            ones_ = tc.ones_like(sort_x)
            cum_sum = (tc.cumsum(sort_x, dim=axis) - scale) / tc.cumsum(ones_, dim=axis)

        mask = sort_x > cum_sum
        mat = tc.arange(mask.numel(), device=x.device, dtype=x.dtype).reshape(mask.shape)
        mask = mask.to(x.dtype) * mat
        if axis is None:
            ind_max = tc.argmax(mask)
        else:
            ind_max = tc.argmax(mask, dim=axis)

        if ind_max.numel() <= 1:
            prox_threshold = cum_sum[ind_max].reshape(scale.shape)
        else:
            if axis is None:
                prox_threshold = tc.minimum(cum_sum[ind_max], x)
            else:
                ind_max_unsq = ind_max.unsqueeze(axis)
                gathered = tc.gather(cum_sum, dim=axis, index=ind_max_unsq)
                prox_threshold = tc.minimum(gathered, x)
                prox_threshold = prox_threshold.reshape(x.shape)

        prox_x = tc.minimum(x, prox_threshold)

        if axis is None:
            if tc.all(mask.bool()):
                prox_x = prox_x * 0
        else:
            all_mask = tc.all(mask.bool(), dim=axis, keepdim=True)
            prox_x = prox_x * (1 - all_mask.to(x.dtype))

        prox_x = prox_x.reshape(sz)
        return prox_x

    def __call__(self, x: tc.Tensor) -> tc.Tensor:
        if self.axis is None:
            return tc.sum(tc.max(x))
        else:
            return tc.sum(tc.max(x, dim=self.axis).values)

    def _check(self, x: tc.Tensor, gamma: tc.Tensor):
        if tc.any(gamma <= 0):
            raise ValueError("'gamma' must be strictly positive")
        if self.axis is None and gamma.numel() > 1:
            raise ValueError("'gamma' must be a scalar when 'axis' is None")
        if gamma.numel() <= 1:
            return
        sz = x.shape
        if len(sz) <= 1:
            self.axis = None
        if len(sz) <= 1:
            raise ValueError("'gamma' must be scalar when 'x' is 1D")
        if len(sz) > 1 and (self.axis is not None):
            sz0 = list(sz)
            sz0[self.axis] = 1
            prod = 1
            for d in sz0: prod *= d
            if gamma.numel() > 1 and (prod != gamma.numel()):
                raise ValueError("Dimension of 'gamma' mismatch")

class Simplex:
    """Compute the projection and the indicator of the simplex."""
    def __init__(self, eta: Union[float, tc.Tensor], axis: Optional[int] = None):
        if not isinstance(eta, tc.Tensor):
            eta = tc.tensor(eta, dtype=tc.float64)
        if tc.any(eta <= 0):
            raise Exception("'eta' must be positive")
        self.eta = eta
        self.axis = axis

    def prox(self, x: tc.Tensor) -> tc.Tensor:
        return x - Max(self.axis).prox(x, self.eta)

    def __call__(self, x: tc.Tensor) -> float:
        if self.axis is None:
            scalar_prod = tc.sum(x)
        else:
            scalar_prod = tc.sum(x, dim=self.axis)
        tol = 1e-10
        if tc.all(x >= 0) and tc.all(tc.abs(scalar_prod - self.eta) < tol):
            return 0
        return float('inf')


def LipschitzSOOT(alpha, beta, eta, N):
    return 1/(alpha*beta) + 1/(2*alpha**2) * max(1, (N*alpha/beta)**2) + 1/eta**2

def gradient_x(x, y, Hmat, alpha, beta, eta, nu):
    l1 = tc.sum(tc.sqrt(x**2 + alpha**2) - alpha, dim=1, keepdim=True)
    l2 = tc.sqrt(tc.sum(x**2 + eta**2, dim=1, keepdim=True))
    Hx_y = tc.matmul(x, Hmat.t()) - y
    gradfid = tc.matmul(Hx_y, Hmat)
    gradl1l2 = nu * (x / (tc.sqrt(x**2 + alpha**2) * (l1 + beta)) - x / (l2**2))
    grad = gradfid + gradl1l2
    return grad, l1

def proj_simplex(x):
    test_simplex = Simplex(eta=1)
    if x.ndim == 2 and x.shape[0] > 1:
        projected = tc.empty_like(x)
        for i in range(x.shape[0]):
            xi = x[i, :].unsqueeze(0)
            projected[i, :] = test_simplex.prox(xi).squeeze(0)
        return projected
    else:
        return test_simplex.prox(x)

def compute_proj_grad_norm(x, gradx, L):
    alpha_val = 1
    temp = x - alpha_val * gradx
    Pgradx = proj_simplex(temp) - x
    return tc.norm(Pgradx)

def majorante_x_diag(x, alpha, l1, beta, Cg2, nu, Hnorm2):
    Al1l2 = nu * (1.0 / (tc.sqrt(x**2 + alpha**2) * (l1 + beta)) + Cg2)
    return Al1l2 + Hnorm2

def Majorante_x(d, x, alpha, l1, beta, Cg2, Hmat, nu):
    Al1l2 = nu * (1.0 / (tc.sqrt(x**2 + alpha**2) * (l1 + beta)) + Cg2)
    Ad = Al1l2 * d + tc.matmul(d, tc.matmul(Hmat.t(), Hmat))
    return Ad

def Criterion(x, Hmat, y, sigma, beta, eta, nu):
    l1 = tc.sum(tc.sqrt(x**2 + sigma**2) - sigma, dim=1, keepdim=True)
    l2 = tc.sqrt(tc.sum(x**2 + eta**2, dim=1, keepdim=True))
    Hx_y = tc.matmul(x, Hmat.t()) - y
    fid = 0.5 * tc.sum(Hx_y**2, dim=1, keepdim=True)
    l1l2 = nu * tc.log((l1 + beta) / l2)
    crit = fid + l1l2
    return crit

def proj_weighted_simplex(z: tc.Tensor, weight: tc.Tensor, n_iter_bisect: int = 60) -> tc.Tensor:
    """Projection ponderee (batchee) sur le simplexe de probabilite.

    Resout, independamment pour chaque ligne du batch :
        min_x  sum_i weight_i/2 * (x_i - z_i)^2
        s.c.   sum_i x_i = 1,  x_i >= 0

    Utilise pour la projection metrique de VMFB (`projection_metric_simplex`
    dans `runVMFBalgorithm.m`), ou `weight` est la majorante diagonale
    (divisee par le pas de relaxation `gamma`) de la Hessienne locale.

    Le multiplicateur de Lagrange associe a la contrainte d'egalite est
    determine par bissection (`n_iter_bisect` iterations, precision
    largement suffisante en double precision), la solution ayant la forme
    fermee x_i = max(z_i - tau/weight_i, 0) pour tau bien choisi (KKT).

    Args:
        z: point a projeter, shape (P, N).
        weight: poids strictement positifs de la metrique diagonale,
            shape (P, N).
        n_iter_bisect: nombre d'iterations de bissection.

    Returns:
        Tenseur projete, shape (P, N).
    """
    zw = z * weight
    # Bissection sur tau : f(tau) = sum_i max(z_i - tau/weight_i, 0) - 1 est
    # decroissante en tau. Bornes garanties valides (cf. docstring interne
    # du depot : f(lo) >= 1 et f(hi) <= 1).
    lo = zw.amin(dim=1, keepdim=True) - weight.amax(dim=1, keepdim=True)
    hi = zw.amax(dim=1, keepdim=True)

    for _ in range(n_iter_bisect):
        mid = 0.5 * (lo + hi)
        x = tc.clamp(z - mid / weight, min=0.0)
        sum_ge_one = x.sum(dim=1, keepdim=True) >= 1.0
        lo = tc.where(sum_ge_one, mid, lo)
        hi = tc.where(sum_ge_one, hi, mid)

    tau = 0.5 * (lo + hi)
    return tc.clamp(z - tau / weight, min=0.0)

def signal_noise(x, xtrue):
    norm1 = tc.sum(tc.abs(x - xtrue), dim=1, keepdim=True) / xtrue.shape[1]
    norm2 = tc.sqrt(tc.sum((x - xtrue)**2, dim=1, keepdim=True) / xtrue.shape[1])
    return norm1, norm2

def display_datas(iter, Crit, l1x, l2x, time_val):
    print("\n----------------------------------")
    print(f"Iteration = {iter}")
    if Crit.ndim == 2: crit_val = Crit[0, 0].item()
    else: crit_val = Crit.item()
    print(f"Crit = {crit_val}")
    print(f"Time = {time_val}")
    print("----------------------------")
    if l1x.ndim == 2: l1_val = l1x[0, 0].item()
    else: l1_val = l1x.item()
    if l2x.ndim == 2: l2_val = l2x[0, 0].item()
    else: l2_val = l2x.item()
    print(f"x : l1  = {l1_val}, l2  = {l2_val}")
    print("----------------------------------")

def dosy_mat(N, M, tmin, tmax, Dmin, Dmax, dtype=tc.float64, device='cpu'):
    D = (tc.log(tc.tensor(Dmin, device=device, dtype=dtype)) -
         tc.log(tc.tensor(Dmax, device=device, dtype=dtype))) / (N - 1)
    T = (tc.tensor(Dmin, device=device, dtype=dtype) *
         tc.exp(-D * tc.arange(1, N + 1, device=device, dtype=dtype)))
    t = tc.linspace(tmin, tmax, M, device=device, dtype=dtype).unsqueeze(1)
    T_row = T.unsqueeze(0)
    kron_t_T = tc.kron(t, T_row)
    Hmat = tc.exp(-kron_t_T)
    return T, Hmat

def phi_prime_cvx(x, delta):
    """Dérivée de la pénalité convexe (ex: L1 lissée)."""
    return x / tc.sqrt(x**2 + delta**2)

def weight_cvx(x, delta):
    """Poids de la matrice majorante pour la pénalité convexe."""
    return 1.0 / tc.sqrt(x**2 + delta**2)

def phi_prime_ncvx(x, delta):
    """Dérivée de la pénalité non-convexe (ex: Cauchy/Geman-McClure)."""
    return x / (1.0 + (x**2) / (delta**2))

def weight_ncvx(x, delta):
    """Poids de la matrice majorante pour la pénalité non-convexe."""
    return 1.0 / (1.0 + (x**2) / (delta**2))

# --- CLASSES DE LOSS (Pour compatibilité nn.Module) ---


class snr_loss(nn.Module):
    """
    -20log(signal/noise)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x_est, x_true):
        batch_snr = []
        
        for xp, xt in zip(x_est, x_true):
            noise = xt - xp
            signal_power = tc.linalg.norm(xt)
            noise_power = tc.linalg.norm(noise)
            snr = 20 * tc.log10(signal_power / (noise_power + 1e-8))
            batch_snr.append(snr)
        
        return -tc.mean(tc.stack(batch_snr))


class tsnr_loss(nn.Module):
    """
    masked snr
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x_est, x_true):
        noise = x_true - x_est
        
        mask = (x_true != 0)
        signal_power = tc.linalg.norm(x_true[mask])
        noise_power = tc.linalg.norm(noise[mask])
        snr = 20 * tc.log10(signal_power / (noise_power + 1e-8))
        
        return -snr
