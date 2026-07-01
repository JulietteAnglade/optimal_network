"""Abstract primitive-function interfaces (algorithm.tex, Section 1).

Every primitive used by the solver is exposed here as an abstract base
class with exactly the methods listed in the document, so that the
functional form (Cobb-Douglas vs. CES production, BPR vs. other
congestion functions, ...) can be swapped without touching any solver
logic (Remark 1).

Notation convention (Section 1.1): for a two-argument primitive
Phi(z1, z2), `grad1`/`grad2` denote the partial derivatives with
respect to the first / second argument respectively, both evaluated at
the point supplied. Single-argument primitives expose a plain `grad`.
"""
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Utility primitives (Section 1.2)
# ---------------------------------------------------------------------------

class UtilityPrimitive(ABC):
    """U(c, l, f_h): at-home utility over shipped consumption, residential
    floor space, and home time."""

    @abstractmethod
    def value(self, c, l, f_h):
        ...

    @abstractmethod
    def grad(self, arg, c, l, f_h):
        """arg in {'c', 'l', 'f_h'}."""
        ...

    @abstractmethod
    def inv_grad(self, arg, target_price, **kwargs):
        """Unique z solving d(Phi)/dz = target_price."""
        ...


class ActivitySubUtilityT(ABC):
    """u_{s'}(d, f) for s' in S^T: activity sub-utility with shipped
    consumption d (delivered tradable good) and floor space f."""

    @abstractmethod
    def value(self, d, f):
        ...

    @abstractmethod
    def grad(self, arg, d, f):
        """arg in {'d', 'f'}."""
        ...

    @abstractmethod
    def inv_grad(self, arg, target_price, **kwargs):
        ...


class ActivitySubUtilityN(ABC):
    """u_{s'}(f) for s' in S^N: activity sub-utility, floor space only."""

    @abstractmethod
    def value(self, f):
        ...

    @abstractmethod
    def grad(self, f):
        ...

    @abstractmethod
    def inv_grad(self, target_price, **kwargs):
        ...


# ---------------------------------------------------------------------------
# Production primitives (Section 1.3)
# ---------------------------------------------------------------------------

class CommercialProductionPrimitive(ABC):
    """F_j^s(H, {L^b}_{b in B^nr}, Q) for s in S^GS: commercial
    goods/services production."""

    @abstractmethod
    def value(self, H, L, Q):
        ...

    @abstractmethod
    def grad(self, arg, H, L, Q):
        """arg in {'H', 'L', 'Q'}."""
        ...

    @abstractmethod
    def inv_grad(self, arg, target_price, **kwargs):
        ...


class ParkingProductionPrimitive(ABC):
    """F_j^p(H, L) for p in S^park: parking-hours production."""

    @abstractmethod
    def value(self, H, L):
        ...

    @abstractmethod
    def grad(self, arg, H, L):
        """arg in {'H', 'L'}."""
        ...

    @abstractmethod
    def inv_grad(self, arg, target_price, **kwargs):
        ...


class TransitVehicleProductionPrimitive(ABC):
    """F^m_{nn'}(H) for m in M^transit: transit vehicle-per-hour
    production."""

    @abstractmethod
    def value(self, H):
        ...

    @abstractmethod
    def grad(self, H):
        ...

    @abstractmethod
    def inv_grad(self, target_price, **kwargs):
        ...


class StationCapacityPrimitive(ABC):
    """F^m_n(l_tilde): station fleet-handling capacity as a function of
    station land. Strictly increasing and concave, invertible derivative."""

    @abstractmethod
    def value(self, l_tilde):
        ...

    @abstractmethod
    def grad(self, l_tilde):
        ...

    @abstractmethod
    def inv_grad(self, target_price, **kwargs):
        ...


# ---------------------------------------------------------------------------
# Network cost and friction primitives (Section 1.4)
# No inversion is required for this group: only value() and partial
# derivatives are ever used.
# ---------------------------------------------------------------------------

class CongestionTime(ABC):
    """T_{nn'}(Qhat_{nn'}, I_{nn'}): road congestion travel time.
    grad1 >= 0 (weakly increasing in flow), grad2 <= 0 (weakly decreasing
    in capacity), jointly convex in Qhat_{nn'}."""

    @abstractmethod
    def value(self, Q_hat, I):
        ...

    @abstractmethod
    def grad1(self, Q_hat, I):
        ...

    @abstractmethod
    def grad2(self, Q_hat, I):
        ...


class IcebergTau(ABC):
    """tau^s_{nn'}(Qhat_{nn'}, I_{nn'}) in (0, 1]: multiplicative iceberg
    transmission factor. grad1 <= 0, grad2 >= 0."""

    @abstractmethod
    def value(self, Q_hat, I):
        ...

    @abstractmethod
    def grad1(self, Q_hat, I):
        ...

    @abstractmethod
    def grad2(self, Q_hat, I):
        ...


class ParkingSearchTime(ABC):
    """T^P_k(y^P_k, S_k) with S_k = sum_{p in P} Y_k^p: parking search
    delay, function of node-level demand and aggregate supply.
    grad1 >= 0, grad2 <= 0."""

    @abstractmethod
    def value(self, y_P, S):
        ...

    @abstractmethod
    def grad1(self, y_P, S):
        ...

    @abstractmethod
    def grad2(self, y_P, S):
        ...


class BoardingTime(ABC):
    """T^{m,board}_{nn'}(Q^{m,B}_{nn'}, V^m_{nn'}): transit boarding delay.
    grad1 >= 0, grad2 <= 0."""

    @abstractmethod
    def value(self, Q_mB, V_m):
        ...

    @abstractmethod
    def grad1(self, Q_mB, V_m):
        ...

    @abstractmethod
    def grad2(self, Q_mB, V_m):
        ...


class WaitingTime(ABC):
    """t^{m,W}_{nn'}(V^m_{nn'}): passenger waiting/in-vehicle time,
    single-argument, weakly decreasing in fleet frequency."""

    @abstractmethod
    def value(self, V_m):
        ...

    @abstractmethod
    def grad(self, V_m):
        ...
