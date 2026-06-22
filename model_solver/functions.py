import numpy as np
from .models import ProductionFunction, UtilityFunction, TauLink, TimeLink

""" class CobbDouglasUtility(UtilityFunction):
    
    U^s(c, l) = sum_k alpha_c[s,k] * log(c_k) + alpha_l[s] * log(l) + alpha_f[s]*log(f).
    alpha_c: (S, S) — alpha_c[s, k] = weight of good k in sector-s utility.
    alpha_l: (S,)
    alpha_f: (S,)
    def __init__(self, alpha_c: np.ndarray, alpha_l: np.ndarray, alpha_f: np.ndarray):
        self.alpha_c = np.asarray(alpha_c, dtype=float)
        self.alpha_l = np.asarray(alpha_l, dtype=float)
        self.alpha_f = np.asarray(alpha_f, dtype=float)
    def __call__(self, c, l, f, s, i=None, j=None):
        c = np.asarray(c)
        return float(np.sum(self.alpha_c[s] * np.log(c)) + self.alpha_l[s] * np.log(l) + self.alpha_f[s] * np.log(f))
    def gradient(self, c, l, f, s, i=None, j=None):
        c = np.asarray(c)
        return self.alpha_c[s] / c, self.alpha_l[s] / l, self.alpha_f[s] / f   # (dU/dc, dU/dl, dU/df)
    def invert(self, p_i, r_i, w_s, s, i=None, j=None):
        c = self.alpha_c[s] / np.maximum(p_i, 1e-8)           # shape (S,)
        l = self.alpha_l[s] / np.maximum(r_i, 1e-8)
        f = self.alpha_f[s] / np.maximum(w_s, 1e-8)
        return c, l, f

class CobbDouglasProduction(ProductionFunction):
  
    Y = A * H^aH * L^aL * prod_k Q_k^{aQ_k}, with sum(a) < 1.
    A:     (S, J)
    aH:    (S,)
    aL:    (S,)
    aQ:    (S, S)  — aQ[s, k] = elasticity of good k in sector-s production.
    def __init__(self, A: np.ndarray, aH: np.ndarray, aL: np.ndarray, aQ: np.ndarray):
        self.A  = np.asarray(A,  dtype=float)
        self.aH = np.asarray(aH, dtype=float)
        self.aL = np.asarray(aL, dtype=float)
        self.aQ = np.asarray(aQ, dtype=float)
        assert np.all(self.aH + self.aL + self.aQ.sum(axis=1) < 1), \
            "Cobb-Douglas requires decreasing returns (sum of elasticities < 1)."
    def __call__(self, H, L, Q_inp, s, j):
        Q_inp = np.asarray(Q_inp, dtype=float)
        # 0**0 = 1 by convention
        prodQ = np.prod(np.where(self.aQ[s] > 0, Q_inp ** self.aQ[s], 1.0))
        return float(self.A[s, j] * H ** self.aH[s] * L ** self.aL[s] * prodQ)
    def gradient(self, H, L, Q_inp, s, j, var=None):
        Y = self(H, L, Q_inp, s, j)
        if var == 'H':     return self.aH[s] * Y / H
        if var == 'L':     return self.aL[s] * Y / L
        if var == 'Q':
            Q_inp = np.asarray(Q_inp, dtype=float)
            return np.where(Q_inp > 0, self.aQ[s] * Y / np.where(Q_inp > 0, Q_inp, 1.0), 0.0)
        # default: return all
        return self.aH[s] * Y / H, self.aL[s] * Y / L, self.aQ[s] * Y / Q_inp
    def invert(self, p_sj, p_vec, w_s, r_j, s, j, H_sj=None):
        aH, aL, aQ = self.aH[s], self.aL[s], self.aQ[s]
        S_sum = aH + aL + aQ.sum()
        p_sj = np.maximum(p_sj, 1e-8)
        w_s = np.maximum(w_s, 1e-8)
        r_j = np.maximum(r_j, 1e-8)
        p_vec = np.maximum(p_vec, 1e-8)
        log_Y  = np.log(self.A[s, j]) + S_sum * np.log(p_sj)
        log_Y += aH * np.log(aH / w_s) + aL * np.log(aL / r_j)
        # 0 * log(0) = 0
        mask = aQ > 0
        log_Y += np.sum(aQ[mask] * np.log(aQ[mask] / p_vec[mask]))
        log_Y /= (1.0 - S_sum)
        log_Y = np.clip(log_Y, -100.0, 100.0)
        Y = np.exp(log_Y)
        H = aH * Y * p_sj / w_s
        L = aL * Y * p_sj / r_j
        Q_inp = np.zeros_like(p_vec)
        Q_inp[mask] = aQ[mask] * Y * p_sj / p_vec[mask]
        return Y, H, L, Q_inp """
    

class CobbDouglasUtility(UtilityFunction):
    """
    Fully vectorized Cobb-Douglas Utility Function.
    Supports operations across all dimensions (S, I, J) simultaneously.
    """
    def __init__(self, alpha_c: np.ndarray, alpha_l: np.ndarray, alpha_f: np.ndarray):
        self.alpha_c = np.asarray(alpha_c, dtype=float)  # (S, K) - weights of goods
        self.alpha_l = np.asarray(alpha_l, dtype=float)  # (S,)   - housing/leisure weight
        self.alpha_f = np.asarray(alpha_f, dtype=float)  # (S,)   - labor preference weight
    def __call__(self, c, l, f, s=None, i=None, j=None):
        """
        Computes utility for all dimensions.
        Inputs: c (K, S, I, I), l (S, I, I), f (S, I, I)
        Output: utility matrix of shape (S, I, I)
        """
        alpha_c_b = self.alpha_c.T[:, :, np.newaxis, np.newaxis]  # (K, S, 1, 1)
        alpha_l_b = self.alpha_l[:, np.newaxis, np.newaxis]       # (S, 1, 1)
        alpha_f_b = self.alpha_f[:, np.newaxis, np.newaxis]       # (S, 1, 1)
        log_c = np.log(np.maximum(c, 1e-12))
        log_l = np.log(np.maximum(l, 1e-12))
        log_f = np.log(np.maximum(f, 1e-12))
        # Sum over the goods dimension (axis 0)
        u_goods = np.sum(alpha_c_b * log_c, axis=0)
        return u_goods + alpha_l_b * log_l + alpha_f_b * log_f
    def gradient(self, c, l, f, s=None, i=None, j=None):
        """
        Computes marginal utilities.
        Outputs: dU_dc (K, S, I, I), dU_dl (S, I, I), dU_df (S, I, I)
        """
        alpha_c_b = self.alpha_c.T[:, :, np.newaxis, np.newaxis]
        alpha_l_b = self.alpha_l[:, np.newaxis, np.newaxis]
        alpha_f_b = self.alpha_f[:, np.newaxis, np.newaxis]
        dU_dc = alpha_c_b / np.maximum(c, 1e-8)
        dU_dl = alpha_l_b / np.maximum(l, 1e-8)
        dU_df = alpha_f_b / np.maximum(f, 1e-8)
        return dU_dc, dU_dl, dU_df
    def invert(self, p, r, w, s=None, i=None, j=None):
        """
        Analytical inversion mapping budget shares directly.
        Inputs: p (K, 1, I, 1), r (1, I, 1), w (S, 1, I)
        Outputs: c (K, S, I, I), l (S, I, I), f (S, I, I)
        """
        alpha_c_b = self.alpha_c.T[:, :, np.newaxis, np.newaxis]
        alpha_l_b = self.alpha_l[:, np.newaxis, np.newaxis]
        alpha_f_b = self.alpha_f[:, np.newaxis, np.newaxis]
        
        c = alpha_c_b / np.maximum(p, 1e-8)
        l = alpha_l_b / np.maximum(r, 1e-8)
        f = alpha_f_b / np.maximum(w, 1e-8)
        return c, l, f

class CobbDouglasProduction(ProductionFunction):
    """
    Fully vectorized Cobb-Douglas Production Function.
    Supports structural operations across all sectors and production zones (S, I).
    """
    def __init__(self, A: np.ndarray, aH: np.ndarray, aL: np.ndarray, aQ: np.ndarray):
        self.A  = np.asarray(A,  dtype=float)  # (S, I)
        self.aH = np.asarray(aH, dtype=float)  # (S,)
        self.aL = np.asarray(aL, dtype=float)  # (S,)
        self.aQ = np.asarray(aQ, dtype=float)  # (S, K)
    def __call__(self, H, L, Q_inp, s=None, j=None):
        """
        Computes production volumes.
        Inputs: H (S, I), L (S, I), Q_inp (S, K, I)
        Output: Y (S, I)
        """
        aH_b = self.aH[:, np.newaxis]      # (S, 1)
        aL_b = self.aL[:, np.newaxis]      # (S, 1)
        aQ_b = self.aQ[:, :, np.newaxis]   # (S, K, 1)
        prodQ = np.prod(np.where(aQ_b > 0, np.maximum(Q_inp, 1e-12) ** aQ_b, 1.0), axis=1)
        Y = self.A * (np.maximum(H, 1e-12) ** aH_b) * (np.maximum(L, 1e-12) ** aL_b) * prodQ
        return Y

    def gradient(self, H, L, Q_inp, s=None, j=None, var=None):
        """
        Computes marginal products.
        Outputs: dH (S, I), dL (S, I), dQ (S, K, I)
        """
        Y = self(H, L, Q_inp)
        aH_b = self.aH[:, np.newaxis]
        aL_b = self.aL[:, np.newaxis]
        aQ_b = self.aQ[:, :, np.newaxis]
        dH = aH_b * Y / np.maximum(H, 1e-8)
        dL = aL_b * Y / np.maximum(L, 1e-8)
        dQ = np.where(aQ_b > 0, aQ_b * Y[:, np.newaxis, :] / np.maximum(Q_inp, 1e-8), 0.0)
        if var == 'H': return dH
        if var == 'L': return dL
        if var == 'Q': return dQ
        return dH, dL, dQ

    def invert(self, p_sj, p_vec, w_s, r_j, H_sj=0, s=None, j=None):
            """
            Analytical cost-minimization inversion.
            Inputs: p_sj (S, I), p_vec (1, K, I), w_s (S, I), r_j (I,) ou (1, I)
            Outputs: Y (S, I), H (S, I), L (S, I), Q_inp (S, K, I)
            """
            S_sum = self.aH + self.aL + self.aQ.sum(axis=1)
            S_sum_b = S_sum[:, np.newaxis]  # (S, 1)
            
            p_sj = np.maximum(p_sj, 1e-8)
            w_s = np.maximum(w_s, 1e-8)
            
            # CORRECTION ROBUSTE : .ravel() garantit un tableau 1D avant le repassage en 2D (1, I)
            r_j_b = np.maximum(r_j.ravel()[np.newaxis, :], 1e-8)  
            p_vec = np.maximum(p_vec, 1e-8)
            
            aH_b = self.aH[:, np.newaxis]
            aL_b = self.aL[:, np.newaxis]
            aQ_b = self.aQ[:, :, np.newaxis]
            
            log_Y = np.log(self.A) + S_sum_b * np.log(p_sj)
            log_Y += np.where(aH_b > 0, aH_b * np.log(np.where(aH_b > 0, aH_b, 1.0) / w_s), 0.0)
            log_Y += np.where(aL_b > 0, aL_b * np.log(np.where(aL_b > 0, aL_b, 1.0) / r_j_b), 0.0)
            
            term_Q = np.where(aQ_b > 0, aQ_b * np.log(np.where(aQ_b > 0, aQ_b, 1.0) / p_vec), 0.0)
            log_Y += term_Q.sum(axis=1)
            
            log_Y /= (1.0 - S_sum_b)
            log_Y = np.clip(log_Y, -100.0, 100.0)
            
            Y = np.exp(log_Y)
            H = aH_b * Y * p_sj / w_s
            L = aL_b * Y * p_sj / r_j_b
            
            Q_inp = np.where(aQ_b > 0, aQ_b * Y[:, np.newaxis, :] * p_sj[:, np.newaxis, :] / p_vec, 0.0)
            return Y, H, L, Q_inp
    
class CobbDouglasCrsProduction(ProductionFunction):
    """
    Y = A * H^aH * L^aL * prod_k Q_k^{aQ_k}, with sum(a) = 1.
    A:     (S, J)
    aH:    (S,)
    aL:    (S,)
    aQ:    (S, S)  — aQ[s, k] = elasticity of good k in sector-s production.
    """
    def __init__(self, A: np.ndarray, aH: np.ndarray, aL: np.ndarray, aQ: np.ndarray):
        self.A   = np.asarray(A,  dtype=float)
        self.aH  = np.asarray(aH, dtype=float)
        self.aL  = np.asarray(aL, dtype=float)
        self.aQ  = np.asarray(aQ, dtype=float)
        assert np.all(self.aH + self.aL + self.aQ.sum(axis=1) == 1)
    
    def __call__(self, H, L, Q_inp, s, j):
        Q_inp = np.asarray(Q_inp, dtype=float)
         # 0**0 = 1 by convention
        prodQ = np.prod(np.where(self.aQ[s] > 0, Q_inp ** self.aQ[s], 1.0))
        return float(self.A[s, j] * H ** self.aH[s] * L ** self.aL[s] * prodQ)
        
    def gradient(self, H, L, Q_inp, s, j, var=None, h=None):
        Y = self(H, L, Q_inp, s, j)
        if var == 'H':     return self.aH[s] * Y / H
        if var == 'L':     return self.aL[s] * Y / L
        if var == 'Q':
            Q_inp = np.asarray(Q_inp, dtype=float)
            return np.where(Q_inp > 0, self.aQ[s] * Y / np.where(Q_inp > 0, Q_inp, 1.0), 0.0)
        # default: return all
        return self.aH[s] * Y / H, self.aL[s] * Y / L, self.aQ[s] * Y / Q_inp
    def invert(self, p_sj, p_vec, w_s, r_j, s, j, H_sj):
        """
        Given prices, wages, rents, and high-skilled labour H_sj, solve for
        optimal Y, H, L, {Q_k} using cost-minimisation under CRS.
            p_sj * aH[s]  * Y / H      = w_s          (FOC H)
            p_sj * aL[s]  * Y / L      = w_s          (FOC L) — note: same wage w_s
            p_sj * aQ[s,k]* Y / Q_k   = p_vec[k]      (FOC Q_k, for every k)
        """
        aH = self.aH[s]
        aL = self.aL[s]
        aQ = self.aQ[s]          # shape (S,)
        A  = self.A[s, j]
        p_vec = np.asarray(p_vec, dtype=float)
        H = float(H_sj)
        L = H * (aL / aH)
        # Intermediates Q_k
        Q = np.where(
            aQ > 0,
            H * (aQ / aH) * (w_s / np.where(p_vec > 0, p_vec, 1.0)),
            0.0
        )
        Y = self(H, L, Q, s, j)
        return Y, H, L, Q
    
class QuadraticProduction(ProductionFunction):
    """
    Y = A_s_j * (alpha_H H + alpha_L L + alpha_Q.Q) - 0.5 * (beta_H H^2 + beta_L L^2 + beta_Q.Q^2)
    """
    def __init__(self, A, alpha_H, alpha_L, alpha_Q, beta_H, beta_L, beta_Q):
        self.A       = np.asarray(A, dtype=float)             # (S, J)
        self.alpha_H = np.asarray(alpha_H, dtype=float)       # (S,)
        self.alpha_L = np.asarray(alpha_L, dtype=float)       # (S,)
        self.alpha_Q = np.asarray(alpha_Q, dtype=float)       # (S, S)
        self.beta_H  = np.asarray(beta_H, dtype=float)        # (S,)
        self.beta_L  = np.asarray(beta_L, dtype=float)        # (S,)
        self.beta_Q  = np.asarray(beta_Q, dtype=float)        # (S, S)
    def __call__(self, H, L, Q_inp, s, j):
        Q = np.asarray(Q_inp)
        lin  = self.alpha_H[s]*H + self.alpha_L[s]*L + (self.alpha_Q[s]*Q).sum()
        quad = self.beta_H[s]*H**2 + self.beta_L[s]*L**2 + (self.beta_Q[s]*Q**2).sum()
        return float(self.A[s, j] * lin - 0.5 * quad)
    def gradient(self, H, L, Q_inp, s, j, var=None):
        Q = np.asarray(Q_inp); A = self.A[s, j]
        if var == 'H': return A*self.alpha_H[s] - self.beta_H[s]*H
        if var == 'L': return A*self.alpha_L[s] - self.beta_L[s]*L
        if var == 'Q': return A*self.alpha_Q[s]  - self.beta_Q[s]*Q
        return None
    def invert(self, p_sj, p_vec, w_s, r_j, s, j, H_sj=None):
        A = self.A[s, j]
        # FOC: p * (A*alpha - beta*x) = price_x  =>  x = (A*alpha - price_x/p) / beta
        H = max(0.0, (A*self.alpha_H[s] - w_s/p_sj) / self.beta_H[s])
        L = max(0.0, (A*self.alpha_L[s] - r_j/p_sj) / self.beta_L[s])
        Q_inp = np.maximum(0.0, (A*self.alpha_Q[s] - p_vec/p_sj) / self.beta_Q[s])
        Y = self(H, L, Q_inp, s, j)
        return Y, H, L, Q_inp

""" class IcebergTau(TauLink):
    
    #tau^s_{nn'} = exp(-delta_s * Qhat_{nn'} / I_{nn'}).
    #delta: scalar or shape (S,)
    #mask: numpy array of shape (n_edges,) with 1 for edges where tau applies, 0 otherwise.
    
    def __init__(self, delta, link_dependance):
        self.delta = np.atleast_1d(np.asarray(delta, dtype=float))  # (S,) or (1,)
        self.link_dependance = link_dependance

    def __call__(self, Q_hat, I):
        mask, tau_scale = self.link_dependance
        ratio = np.ravel(Q_hat) / np.ravel(I) 
        return mask * np.exp(-self.delta[:, None] * tau_scale[None, :] * ratio[None, :])

    def gradient(self, Q_hat, I, var):
        tau = self(Q_hat, I)                           # (S, n_edges)
        mask, tau_scale = self.link_dependance
        
        Q_hat_flat = np.ravel(Q_hat)
        I_flat = np.ravel(I)
        
        I_safe = np.maximum(I_flat[None, :], 1e-8)
        
        if var == 'Q':
            return - mask[None, :] * self.delta[:, None] / I_safe * tau
        if var == 'I':
            return mask * self.delta[:, None] * tau_scale[None, :] * Q_hat_flat[None, :] / (I_safe ** 2) * tau
            
        raise ValueError(f"unknown var={var}") """


class IcebergTau(TauLink):
    """
    tau^s_e = exp(-delta_s * tau_scale_e * (1 + a*(Qhat_e/I_e)^b))

    delta: (S,)
    a,b: congestion parameters (same as BPRTime)
    link_dependance: (mask, tau_scale)
    """

    def __init__(self, delta, link_dependance, a=0.6, b=6.0):
        self.delta = np.atleast_1d(np.asarray(delta, dtype=float))
        self.link_dependance = link_dependance
        self.a = float(a)
        self.b = float(b)

    def __call__(self, Q_hat, I):
        mask, tau_scale = self.link_dependance

        Q = np.ravel(Q_hat)
        I = np.maximum(np.ravel(I), 1e-8)

        ratio = np.clip(Q / I, 0.0, 10.0)
        congestion = 1.0 + self.a * ratio**self.b

        tau = mask[None, :] * np.exp(
            -self.delta[:, None] * tau_scale[None, :] * congestion[None, :]
        )

        return tau

    def gradient(self, Q_hat, I, var):
        tau = self(Q_hat, I)

        mask, tau_scale = self.link_dependance

        Q = np.ravel(Q_hat)
        I = np.maximum(np.ravel(I), 1e-8)

        ratio = np.clip(Q / I, 0.0, 10.0)

        common = (
            self.delta[:, None]
            * tau_scale[None, :]
            * self.a
            * self.b
            * ratio[None, :]**(self.b - 1)
        )

        if var == 'Q':
            return -mask[None, :] * common * tau / I[None, :]

        if var == 'I':
            return mask[None, :] * common * tau * Q[None, :] / (I[None, :]**2)

        raise ValueError(f"unknown var={var}")


class BPRTime(TimeLink):
    """
    T_{nn'} = T0 * (1 + a * (Qhat / I)^b).
    T0:   shape (n_edges,)
    a, b: scalars (typically a=0.15, b=4).
    """
    def __init__(self, T0, a: float = 0.6, b: float = 6.0):
        self.T0 = np.asarray(T0, dtype=float)          # (n_edges,)
        self.a  = float(a)
        self.b  = float(b)
    def __call__(self, Q_hat, I):
        ratio = np.divide(Q_hat, I, out=np.zeros_like(Q_hat), where=I != 0)
        ratio = np.clip(ratio, 0.0, 10.0)
        return self.T0 * (1.0 + self.a * ratio ** self.b)
    def gradient(self, Q_hat, I, var):
        ratio = np.divide(Q_hat, I, out=np.zeros_like(Q_hat), where=I != 0)
        ratio = np.clip(ratio, 0.0, 20.0)
        if var == 'Q':
            return self.T0 * self.a * self.b * ratio ** (self.b - 1) / np.maximum(I, 1e-8)
        if var == 'I':
            return -self.T0 * self.a * self.b * ratio ** self.b / np.maximum(I, 1e-8)
        raise ValueError(f"unknown var={var}")