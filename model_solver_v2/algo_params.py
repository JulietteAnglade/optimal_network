"""Algorithmic hyperparameters: step sizes, loop caps, tolerances, and the
simulated-annealing / multi-start schedule for I_{nn'} (Section 2, eq. 1-4
and Section 5.1 step 4(a))."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AlgoParams:
    # ---- Inner loop (eq. 1-4) ----
    T_inner: int = 30           # max inner iterations per outer cycle
    alpha_x: float = 0.05       # primal ascent step (eq. 3, 32-41)
    alpha_lambda: float = 0.05  # dual descent step (eq. 4, 42-60)
    tol_inner: float = 1e-3     # eps_int, inner convergence tolerance

    # ---- Routing mode (routing.py) ----
    # "discrete": exact, as literally specified by the document -- heap
    #   Dijkstra (Section 3.1) and hard-max Bellman-Ford-Iceberg (Section 3.2),
    #   with all-or-nothing path flow assignment.
    # "continuous": an entropy-regularized (soft-min / soft-max) relaxation
    #   of the same recursions, with a logit-weighted fractional flow
    #   loading instead of all-or-nothing assignment. This is a stabilizing
    #   extension beyond the literal document -- the smoothing keeps route
    #   choice (and hence link flows, and hence the FONC residuals feeding
    #   the gradient steps) continuous in the background prices instead of
    #   jumping discontinuously between shortest paths, which can damp
    #   oscillation in the primal-dual inner loop. routing_temperature is
    #   the smoothing parameter sigma (-> 0 recovers the discrete solution).
    routing_mode: str = "discrete"        # "discrete" | "continuous"
    routing_temperature: float = 1.0      # sigma for the continuous mode
    routing_n_iter: Optional[int] = None  # relaxation/Bellman-Ford rounds; default = n_nodes

    # ---- Intermediate loop (MSA fixed point on Z, eq. 61) ----
    T_msa: int = 200             # max intermediate iterations per outer cycle
    tol_msa: float = 1e-3       # MSA convergence tolerance on Z's change

    # ---- Outer loop ----
    T_outer: int = 15           # max outer cycles
    tol_outer: float = 1e-3     # eps_ext, outer convergence tolerance

    # ---- Outer structural primal step sizes (Table 2) ----
    alpha_I: float = 0.05       # infrastructure gradient step (eq. 62-63)
    alpha_l: float = 0.05       # floor-space gradient step (eq. 64)
    alpha_Y: float = 0.05       # parking-hours supply gradient step (eq. 66)
    alpha_V: float = 0.05       # transit vehicle flow gradient step (eq. 68)
    alpha_beta: float = 0.05    # capital-budget shadow price step (eq. 74)

    # ---- Simulated annealing for I_{nn'} (eq. 62-63) ----
    sa_T_init: float = 1.0
    sa_cooling: float = 0.95    # c_cool, geometric cooling T_{K+1} = c_cool * T_K
    sa_multistart_every: int = 5  # restart from a random I_{nn'} config every N outer cycles
    sa_multistart_count: int = 3  # number of random restarts to try at each multi-start event

    # ---- Diminishing step-size schedule ----
    # Constant step sizes only give bounded oscillation around a solution
    # for this kind of primal-dual/subgradient scheme, not convergence to
    # it; a decaying schedule (satisfying the usual Robbins-Monro-style
    # sum-diverges/sum-of-squares-converges conditions for "sqrt") is the
    # standard fix. "linear" decays faster (also satisfies those
    # conditions but discards old progress quicker); None disables decay
    # (matches the previous constant-step-size behavior).
    decay_type: Optional[str] = "sqrt"  # "sqrt" | "linear" | None
    decay_start: int = 0                 # iteration/cycle index decay begins at

    def eta(self, eta_base: float, t: int) -> float:
        """Effective step size at iteration/cycle t, given decay_type."""
        if self.decay_type is None or t < self.decay_start:
            return eta_base
        dt = t - self.decay_start + 1
        if self.decay_type == "sqrt":
            return eta_base / np.sqrt(dt)
        if self.decay_type == "linear":
            return eta_base / dt
        raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def msa_weight(self, K: int) -> float:
        """theta_K = 1 / K (MSA damping weight at outer cycle K, K >= 1)."""
        return 1.0 / max(1, K)
