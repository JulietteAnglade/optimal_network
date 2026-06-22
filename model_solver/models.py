import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class Functions(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass


class ProductionFunction(Functions):
    @abstractmethod
    def invert(*args, **kwargs):
        """Given prices p, compute the inputs H, L, Q_inp."""
        pass


class UtilityFunction(Functions):
    @abstractmethod
    def invert(*args, **kwargs):
        """Given prices, compute the consumption c,  land l and leisure time."""
        pass


class TauLink(Functions):
    pass


class TimeLink(Functions):
    pass

class WaitingTimeLink(Functions):
    pass

class ParkingSearchTimeNode(Functions):
    pass

class BoardingTimeLink(Functions):
    pass

#check whether we need to add other class for other functions


@dataclass
class Network:
    """Transport network: nodes and edges."""
    n_zones: Optional[int] = None
    n_nodes: Optional[int] = None
    edges: Optional[List[Tuple[int, int]]] = None
    n_modes: int = 1
    n_edges: Optional[int] = None
    kappa: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    I_min: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    I_max: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    link_dependance_tau: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) #à supprimer, la dépendance en link sera gérée directement dans la fonction tau_link
    link_dependance_t: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) #pareil à supprimer, la dépendance en link sera gérée directement dans la fonction t_link
    zone_to_nodes: Optional[np.ndarray] = None # à unifier avec la variable mode_to_nodes, pour avoir un seul mapping zone -> node par mode
    zone_to_node: Optional[np.ndarray] = None
    mode_to_nodes: Optional[np.ndarray] = None
    original_edges: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # adjacency lists built after init
    adj_out: List[List[Tuple[int, int]]] = field(init=False)
    adj_in: List[List[Tuple[int, int]]] = field(init=False)

    def __post_init__(self):
        self.adj_out = [[] for _ in range(self.n_nodes)]
        self.adj_in = [[] for _ in range(self.n_nodes)]
        if self.edges is not None:
            for edge_idx, (u, v) in enumerate(self.edges):
                if 0 <= u < self.n_nodes and 0 <= v < self.n_nodes:
                    self.adj_out[u].append((v, edge_idx))
                    self.adj_in[v].append((u, edge_idx))
                else:
                    raise ValueError(f"Edge {(u, v)} references node outside [0, {self.n_nodes})")


@dataclass
class ModelParams:
    #several parameters need to be added to align to the new enriched model --> add them
    """Economic model parameters."""
    # ---- Sizes ----
    n_zones: int                                  # |I|
    n_sectors: int                                # |S|

    # ---- Production ----
    production_function: ProductionFunction  # function to compute output Y given inputs H, L, Q_inp

    # ---- Households ----
    utility_function: UtilityFunction  # function to utility U given consumption c and land l

    # ---- Endowments ----
    L_bar: np.ndarray                             # land per zone, shape (J,)
    H_bar: float                                  # total time per capita
    q_bar: float                                  # total population
    K: float                                      # infrastructure budget

    # ---- Iceberg costs ---
    sector_depedance_tau: Optional[np.array]
    tau_link: TauLink  # per-edge iceberg cost, shape (n_edges,) given infrastructure I_infra and flow Qhat

    # ---- Transportation time ----
    t_link: TimeLink  # per-edge transport time, shape (n_edges,) given infrastructure I_infra and flow Qhat

    # ---- Transport ----
    d_w: float                                    # working days (time -> cost conversion)
    phi: np.ndarray                               # weight of good s per unit volume, shape (S,)

    # ---- Logit ----
    sigma: np.ndarray                             # OD logit,     shape (S,)
    sigma_tilde: float                            # sector logit
    
    #---- Network ----
    network: Network


@dataclass
class ModelState:
    """Current state: prices, quantities, flows."""
    # Several variables need to be added to align with the new enriched model 
    # ---- Prices (outer loop) ----
    p: np.ndarray                                 # goods prices,    shape (S, J)
    w: np.ndarray                                 # wages,           shape (S, J)
    r: np.ndarray                                 # land rents,      shape (J,)
    beta: float                                   # infrastructure price
    gamma: np.ndarray                             # congestion,      shape (n_edges,)
   

    # ---- Production / households ----
    Y: np.ndarray                                 # output,          shape (S, J)
    H_prod: np.ndarray                            # labor demand,    shape (S, J)
    L_prod: np.ndarray                            # land demand,     shape (S, J)
    Q_inter: np.ndarray                           # intersectoral inputs, shape (S, S, J)

    c: np.ndarray                                 # consumption, shape (S, S, I, J)
    l_res: np.ndarray                             # residential land, shape (S, I, J)
    f : np.ndarray                                # leisure time, shape (S, I, J)

    
    # ---- Flows (inner loop) ----
    Q_ship: np.ndarray                            # shipped volumes Q^s_{ij}, shape (S, I, J)
    Q_tilde: np.ndarray                           # received volumes,         shape (S, I, J)
    tau_eff: np.ndarray                           # effective retention factor on route, shape (S, I, J)
    active: np.ndarray                            # active OD arcs mask, shape (S, I, J)

    q_pop: np.ndarray                             # population q^s_{ij}, shape (S, I, J)
    q_bar_s: np.ndarray                           # population by sector, shape (S,)

    Q_hat: np.ndarray                             # total flow per edge, shape (n_edges,)
    q_hat_sj: np.ndarray                           # commuters by sector,   shape (S, I, n_edges)
    Q_hat_sj_goods: np.ndarray                     # goods by sector,       shape (S, I, n_edges)
    Q_hat_sj_price_goods: np.ndarray               # price-weighted goods,  shape (S, I, n_edges) = sum_sj nu[s,i,j] * Q_goods_link[s,i,j]

    # ---- Infrastructure ----
    I_infra: np.ndarray                           # infrastructure levels, shape (n_edges,)
    gamma_I: np.ndarray                           # infra gradient,  shape (n_edges,)


@dataclass
class AlgoParams:
    #Need to figure out if adding new hyperparameters is necessary for the new enriched model
    # Outer loop
    T_outer: int = 200
    eta_beta: float = 0.01
    eta_I: float = 0.01

    #Allocation loop
    T_allocation: float = 200
    eta_p: float = 0.05
    eta_w: float = 0.05
    eta_r: float = 0.05
    eta_H: float = 0.01

    # Inner loop (flows)
    T_inner: int = 50
    alpha_Q: float = 0.2                          # volume update step
    alpha_Qhat: float = 0.3                       # MSA damping on Qhat
    eta_gamma: float = 0.01

    # Tolerances
    tol_outer: float = 1e-4
    tol_inner: float = 1e-4

    # Logging
    verbose: bool = True
    log_every: int = 10

    # Step decay
    decay_start: int = 0
    decay_type: str = "sqrt"


    #Here simulated annealing and MSA needs to be more cleary coded
    #Check the code of simulated annealing
    # Check the code of MSA --> maybe replace linear decay by MSA as MSA will not be needed anymore. 
    # Simulated Annealing
    sa_enabled: bool = False
    sa_T_init : float = 5.0 #intial temperature
    sa_alpha: float = 0.98 #cooling rate
    sa_noise_scale: float = 0.1 #scale of the noise added to the gradients

    def sa_temperature(self, t: int) -> float:
        """Return the simulated annealing temperature at iteration t."""
        if not self.sa_enabled:
            return 0.0
        return self.sa_T_init * (self.sa_alpha ** t)

    def eta(self, eta_base: float, t: int) -> float:
        """Return the effective step size at iteration t."""
        if self.decay_type == None or t < self.decay_start:
            return eta_base
        dt = t - self.decay_start + 1
        if self.decay_type == "sqrt":
            return eta_base / np.sqrt(dt)
        if self.decay_type == "linear":
            return eta_base / dt
        raise ValueError(f"Unknown decay_type: {self.decay_type}")
    
