"""Concrete example functional forms for every abstract primitive in
primitives.py.

The document (Remark 1) intentionally leaves functional forms abstract;
these are reasonable concrete instantiations (Cobb-Douglas utility and
production, BPR-style congestion/boarding/parking-search/waiting
frictions, an exponential-congestion iceberg factor) so that the solver
in model_solver_v2 is runnable end-to-end. Swapping any of these for an
alternative form requires no change to routing.py / inner_loop.py /
outer_loop.py, only to params.py wiring.

Array-shape convention: every primitive here is vectorized over the
relevant index sets at once (e.g. commercial production over sectors S
and zones I simultaneously), matching the shapes used in state.py.
A floor of 1e-8 is applied before any division/log to avoid singularities,
mirroring model_solver/functions.py's clipping style.
"""
import numpy as np

from .primitives import (
    UtilityPrimitive, ActivitySubUtilityT, ActivitySubUtilityN,
    CommercialProductionPrimitive, ParkingProductionPrimitive,
    TransitVehicleProductionPrimitive, StationCapacityPrimitive,
    CongestionTime, IcebergTau, ParkingSearchTime, BoardingTime, WaitingTime,
)

_EPS = 1e-8


def _pos(x):
    return np.maximum(np.asarray(x, dtype=float), _EPS)


# ---------------------------------------------------------------------------
# Utility primitives
# ---------------------------------------------------------------------------

class CobbDouglasUtility(UtilityPrimitive):
    """U(c, l, f_h) = sum_{s' in S^T} alpha_c[s'] log(c_{s'}) + alpha_l log(l)
    + alpha_fh log(f_h). Single shared instance (the document's U takes no
    (i,j,s,b) arguments), broadcasts over any leading batch dimensions."""

    def __init__(self, alpha_c: np.ndarray, alpha_l: float, alpha_fh: float):
        self.alpha_c = np.asarray(alpha_c, dtype=float)  # (n_tradable,)
        self.alpha_l = float(alpha_l)
        self.alpha_fh = float(alpha_fh)

    def value(self, c, l, f_h):
        c = _pos(c)
        return np.sum(self.alpha_c * np.log(c), axis=-1) + self.alpha_l * np.log(_pos(l)) + self.alpha_fh * np.log(_pos(f_h))

    def grad(self, arg, c, l, f_h):
        if arg == 'c':
            return self.alpha_c / _pos(c)
        if arg == 'l':
            return self.alpha_l / _pos(l)
        if arg == 'f_h':
            return self.alpha_fh / _pos(f_h)
        raise ValueError(f"unknown arg={arg}")

    def inv_grad(self, arg, target_price, **kwargs):
        target_price = _pos(target_price)
        if arg == 'c':
            return self.alpha_c / target_price
        if arg == 'l':
            return self.alpha_l / target_price
        if arg == 'f_h':
            return self.alpha_fh / target_price
        raise ValueError(f"unknown arg={arg}")


class CobbDouglasActivityUtilityT(ActivitySubUtilityT):
    """u_{s'}(d, f) = alpha_d[s'] log(d) + alpha_f[s'] log(f), s' in S^T,
    vectorized over the tradable-activity axis (last axis of d, f)."""

    def __init__(self, alpha_d: np.ndarray, alpha_f: np.ndarray):
        self.alpha_d = np.asarray(alpha_d, dtype=float)
        self.alpha_f = np.asarray(alpha_f, dtype=float)

    def value(self, d, f):
        return self.alpha_d * np.log(_pos(d)) + self.alpha_f * np.log(_pos(f))

    def grad(self, arg, d, f):
        if arg == 'd':
            return self.alpha_d / _pos(d)
        if arg == 'f':
            return self.alpha_f / _pos(f)
        raise ValueError(f"unknown arg={arg}")

    def inv_grad(self, arg, target_price, **kwargs):
        target_price = _pos(target_price)
        if arg == 'd':
            return self.alpha_d / target_price
        if arg == 'f':
            return self.alpha_f / target_price
        raise ValueError(f"unknown arg={arg}")


class CobbDouglasActivityUtilityN(ActivitySubUtilityN):
    """u_{s'}(f) = alpha_f[s'] log(f), s' in S^N."""

    def __init__(self, alpha_f: np.ndarray):
        self.alpha_f = np.asarray(alpha_f, dtype=float)

    def value(self, f):
        return self.alpha_f * np.log(_pos(f))

    def grad(self, f):
        return self.alpha_f / _pos(f)

    def inv_grad(self, target_price, **kwargs):
        return self.alpha_f / _pos(target_price)


# ---------------------------------------------------------------------------
# Production primitives
# ---------------------------------------------------------------------------

class CobbDouglasCommercialProduction(CommercialProductionPrimitive):
    """F_j^s(H, L, Q) = A[s,j] * H^aH[s] * prod_b L_b^aL[s,b] * prod_k Q_k^aQ[s,k],
    decreasing returns (sum of elasticities < 1). Vectorized over (S, I).

    inv_grad uses the production-identity shortcut explicit in the document
    ("inverting local marginal productivity conditions for a given
    production target Y_j^{s,t}"): since d F/d z = a_z * Y / z for a
    Cobb-Douglas, inv_grad(arg, target_price, Y) returns a_z * Y / target_price.
    """

    def __init__(self, A: np.ndarray, aH: np.ndarray, aL: np.ndarray, aQ: np.ndarray):
        self.A = np.asarray(A, dtype=float)    # (S, I)
        self.aH = np.asarray(aH, dtype=float)  # (S,)
        self.aL = np.asarray(aL, dtype=float)  # (S, B_nr)
        self.aQ = np.asarray(aQ, dtype=float)  # (S, n_tradable)

    def value(self, H, L, Q):
        aH = self.aH[:, None]
        aL = self.aL[:, None, :]
        aQ = self.aQ[:, None, :]
        prodL = np.prod(np.where(aL > 0, _pos(L) ** aL, 1.0), axis=-1)
        prodQ = np.prod(np.where(aQ > 0, _pos(Q) ** aQ, 1.0), axis=-1)
        return self.A * _pos(H) ** aH * prodL * prodQ

    def grad(self, arg, H, L, Q):
        Y = self.value(H, L, Q)
        if arg == 'H':
            return self.aH[:, None] * Y / _pos(H)
        if arg == 'L':
            return self.aL[:, None, :] * Y[:, :, None] / _pos(L)
        if arg == 'Q':
            return self.aQ[:, None, :] * Y[:, :, None] / _pos(Q)
        raise ValueError(f"unknown arg={arg}")

    def inv_grad(self, arg, target_price, Y):
        Y = np.asarray(Y, dtype=float)
        target_price = _pos(target_price)
        if arg == 'H':
            return self.aH[:, None] * Y / target_price
        if arg == 'L':
            return self.aL[:, None, :] * Y[:, :, None] / target_price
        if arg == 'Q':
            return self.aQ[:, None, :] * Y[:, :, None] / target_price
        raise ValueError(f"unknown arg={arg}")


class CobbDouglasParkingProduction(ParkingProductionPrimitive):
    """F_j^p(H, L) = A[j,p] * H^aH[p] * L^aL[p], vectorized over (I, n_p)."""

    def __init__(self, A: np.ndarray, aH: np.ndarray, aL: np.ndarray):
        self.A = np.asarray(A, dtype=float)    # (I, P)
        self.aH = np.asarray(aH, dtype=float)  # (P,)
        self.aL = np.asarray(aL, dtype=float)  # (P,)

    def value(self, H, L):
        return self.A * _pos(H) ** self.aH * _pos(L) ** self.aL

    def grad(self, arg, H, L):
        Y = self.value(H, L)
        if arg == 'H':
            return self.aH * Y / _pos(H)
        if arg == 'L':
            return self.aL * Y / _pos(L)
        raise ValueError(f"unknown arg={arg}")

    def inv_grad(self, arg, target_price, Y):
        Y = np.asarray(Y, dtype=float)
        target_price = _pos(target_price)
        if arg == 'H':
            return self.aH * Y / target_price
        if arg == 'L':
            return self.aL * Y / target_price
        raise ValueError(f"unknown arg={arg}")


class CobbDouglasTransitVehicleProduction(TransitVehicleProductionPrimitive):
    """F^m_{nn'}(H) = A[m,e] * H^a[m], vectorized over (M, n_edges)."""

    def __init__(self, A: np.ndarray, a: np.ndarray):
        self.A = np.asarray(A, dtype=float)  # (M, n_edges)
        self.a = np.asarray(a, dtype=float)  # (M,)

    def value(self, H):
        return self.A * _pos(H) ** self.a[:, None]

    def grad(self, H):
        Y = self.value(H)
        return self.a[:, None] * Y / _pos(H)

    def inv_grad(self, target_price, Y):
        Y = np.asarray(Y, dtype=float)
        return self.a[:, None] * Y / _pos(target_price)


class LogStationCapacity(StationCapacityPrimitive):
    """F^m_n(l_tilde) = A[m] * log(1 + l_tilde): strictly increasing and
    concave, with a directly invertible derivative F' = A / (1 + l_tilde)
    (no production-target shortcut needed for a single-argument primitive)."""

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, dtype=float)  # (M,)

    def value(self, l_tilde):
        return self.A[:, None] * np.log1p(np.maximum(l_tilde, 0.0))

    def grad(self, l_tilde):
        return self.A[:, None] / (1.0 + np.maximum(l_tilde, 0.0))

    def inv_grad(self, target_price, **kwargs):
        target_price = _pos(target_price)
        return np.maximum(self.A[:, None] / target_price - 1.0, 0.0)


# ---------------------------------------------------------------------------
# Network cost and friction primitives (value + grad1/grad2 only)
# ---------------------------------------------------------------------------

class BPRCongestionTime(CongestionTime):
    """T_{nn'} = T0 * (1 + a * (Qhat / I)^b)."""

    def __init__(self, T0: np.ndarray, a: float = 0.6, b: float = 6.0):
        self.T0 = np.asarray(T0, dtype=float)  # (n_edges,)
        self.a = float(a)
        self.b = float(b)

    def _ratio(self, Q_hat, I):
        return np.clip(np.asarray(Q_hat, dtype=float) / _pos(I), 0.0, 10.0)

    def value(self, Q_hat, I):
        return self.T0 * (1.0 + self.a * self._ratio(Q_hat, I) ** self.b)

    def grad1(self, Q_hat, I):
        r = self._ratio(Q_hat, I)
        return self.T0 * self.a * self.b * r ** (self.b - 1.0) / _pos(I)

    def grad2(self, Q_hat, I):
        r = self._ratio(Q_hat, I)
        return -self.T0 * self.a * self.b * r ** self.b / _pos(I)


class ExponentialIcebergTau(IcebergTau):
    """tau^s_{nn'} = mask_e * exp(-delta[s] * tau_scale[e] * (1 + a * (Qhat/I)^b))."""

    def __init__(self, delta: np.ndarray, mask: np.ndarray, tau_scale: np.ndarray, a: float = 0.6, b: float = 6.0):
        self.delta = np.asarray(delta, dtype=float)          # (S,)
        self.mask = np.asarray(mask, dtype=float)             # (n_edges,)
        self.tau_scale = np.asarray(tau_scale, dtype=float)   # (n_edges,)
        self.a = float(a)
        self.b = float(b)

    def _ratio(self, Q_hat, I):
        return np.clip(np.asarray(Q_hat, dtype=float) / _pos(I), 0.0, 10.0)

    def _congestion(self, Q_hat, I):
        return 1.0 + self.a * self._ratio(Q_hat, I) ** self.b

    def value(self, Q_hat, I):
        cong = self._congestion(Q_hat, I)
        return self.mask[None, :] * np.exp(-self.delta[:, None] * self.tau_scale[None, :] * cong[None, :])

    def grad1(self, Q_hat, I):
        tau = self.value(Q_hat, I)
        r = self._ratio(Q_hat, I)
        d_cong_dQ = self.a * self.b * r ** (self.b - 1.0) / _pos(I)
        return -self.delta[:, None] * self.tau_scale[None, :] * d_cong_dQ[None, :] * tau

    def grad2(self, Q_hat, I):
        tau = self.value(Q_hat, I)
        r = self._ratio(Q_hat, I)
        d_cong_dI = -self.a * self.b * r ** self.b / _pos(I)
        return -self.delta[:, None] * self.tau_scale[None, :] * d_cong_dI[None, :] * tau


class PowerParkingSearchTime(ParkingSearchTime):
    """T^P_k(y^P, S) = c0[k] * (y^P / S)^b, increasing in demand y^P,
    decreasing in aggregate supply S."""

    def __init__(self, c0: np.ndarray, b: float = 1.0):
        self.c0 = np.asarray(c0, dtype=float)  # (n_zones,)
        self.b = float(b)

    def value(self, y_P, S):
        ratio = np.asarray(y_P, dtype=float) / _pos(S)
        return self.c0 * ratio ** self.b

    def grad1(self, y_P, S):
        ratio = np.asarray(y_P, dtype=float) / _pos(S)
        return self.c0 * self.b * ratio ** (self.b - 1.0) / _pos(S)

    def grad2(self, y_P, S):
        ratio = np.asarray(y_P, dtype=float) / _pos(S)
        return -self.c0 * self.b * ratio ** self.b / _pos(S)


class BPRBoardingTime(BoardingTime):
    """T^{m,board}_{nn'}(Q^{m,B}, V^m) = T0 * (1 + a * (Q^{m,B} / V^m)^b)."""

    def __init__(self, T0: np.ndarray, a: float = 0.6, b: float = 4.0):
        self.T0 = np.asarray(T0, dtype=float)  # (M, n_edges)
        self.a = float(a)
        self.b = float(b)

    def _ratio(self, Q_mB, V_m):
        return np.clip(np.asarray(Q_mB, dtype=float) / _pos(V_m), 0.0, 10.0)

    def value(self, Q_mB, V_m):
        return self.T0 * (1.0 + self.a * self._ratio(Q_mB, V_m) ** self.b)

    def grad1(self, Q_mB, V_m):
        r = self._ratio(Q_mB, V_m)
        return self.T0 * self.a * self.b * r ** (self.b - 1.0) / _pos(V_m)

    def grad2(self, Q_mB, V_m):
        r = self._ratio(Q_mB, V_m)
        return -self.T0 * self.a * self.b * r ** self.b / _pos(V_m)


class PowerWaitingTime(WaitingTime):
    """t^{m,W}_{nn'}(V^m) = c0 / (V^m)^b, decreasing in fleet frequency."""

    def __init__(self, c0: np.ndarray, b: float = 1.0):
        self.c0 = np.asarray(c0, dtype=float)  # (M, n_edges)
        self.b = float(b)

    def value(self, V_m):
        return self.c0 / _pos(V_m) ** self.b

    def grad(self, V_m):
        return -self.b * self.c0 / _pos(V_m) ** (self.b + 1.0)
