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
    #define modes ? public transit modes, car ?
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
    #réfléchir à comment ajouter les différences entre non-tradable, non tradable etc.
    n_b_r:int                                           # |B^r|, number of residential land types
    n_b_nr:int                                           # |B^nr|, number of residential land types
    n_p: int                                          # |P|, number of parking types (eg. garage, street, underground)
    #est-ce que c'est nécessaire de définir les ensembles explicitement

    # ---- Production ----
    production_function: ProductionFunction  # function to compute output Y given inputs H, L, Q_inp
    vehicle_production_function: ProductionFunction  # function to compute output Y given inputs H_{nn'}^m
    parking_production_function: ProductionFunction  # function to compute output Y given H, L, Q
    fleet_production_function: ProductionFunction  # function to compute output Y given H, L

    # ---- Households ----
    utility_function: UtilityFunction  # function to utility U given consumption c and land l
    utility_function_bis_s : UtilityFunction  # function to utility u given for all GS sector s, given and f or f only --> est-ce qu'il faut une autre classe de fonction ?

    # ---- Endowments ----
    L_bar: np.ndarray                             # land per zone, shape (J,)
    H_bar: float                                  # total time per capita
    q_bar: float                                  # total population
    K: float                                      # infrastructure budget
    q_bar_0: float                                # peripheral labor pool capacity
    V_m : np.ndarray                              # fleet size per mode, shape (M,)


    # ---- Iceberg costs ---
    tau_link: TauLink  # per-edge and per sectoriceberg cost, shape (n_edges, n_sectors) given infrastructure I_infra and flow Qhat

    # ---- Transportation time ----
    t_link: TimeLink  # per-edge transport time, shape (n_edges,) given infrastructure I_infra and flow Qhat

    # Parking search time
    t_park_node: ParkingSearchTimeNode  # per-node parking search time, shape (n_zones,) given parking hours demand y^P and parking hours supply sum Y^p

    # Boarding time
    t_board_link: BoardingTimeLink  # per-edge boarding time, shape (n_edges, n_modes) given in

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

    # --- routing potentials ---
    mu_commuter: np.ndarray = field(default_factory=lambda: np.array([]))
    nu_goods: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- Multimodal / parking / boarding outer variables ----
    y_k_P: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_m_B: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_park: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma_board: np.ndarray = field(default_factory=lambda: np.array([]))
    V_m: np.ndarray = field(default_factory=lambda: np.array([]))
    w_m: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_fleet_m: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_v_cons_mn: np.ndarray = field(default_factory=lambda: np.array([]))
    lambda_stat_mn: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- Household / production extended variables ----
    h_ijsb: np.ndarray = field(default_factory=lambda: np.array([]))
    l_bar_b: np.ndarray = field(default_factory=lambda: np.array([]))
    l_tilde_b: np.ndarray = field(default_factory=lambda: np.array([]))
    q0_js: np.ndarray = field(default_factory=lambda: np.array([]))
    h0_js: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_i0: np.ndarray = field(default_factory=lambda: np.array([]))
    Q_0i: np.ndarray = field(default_factory=lambda: np.array([]))
    l_tilde_infra: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- dual prices and analytical shadow values ----
    lambda0: float = 0.0
    w0: float = 0.0
    rho_js: np.ndarray = field(default_factory=lambda: np.array([]))
    r_tilde_i: np.ndarray = field(default_factory=lambda: np.array([]))
    r_bar_b: np.ndarray = field(default_factory=lambda: np.array([]))
    r_b_r: np.ndarray = field(default_factory=lambda: np.array([]))
    r_j_b_nr: np.ndarray = field(default_factory=lambda: np.array([]))
    r_j_p: np.ndarray = field(default_factory=lambda: np.array([]))
    r_i_infra: np.ndarray = field(default_factory=lambda: np.array([]))
    pi_j_s: np.ndarray = field(default_factory=lambda: np.array([]))
    pi_j_p: np.ndarray = field(default_factory=lambda: np.array([]))
    pi_nn_m: np.ndarray = field(default_factory=lambda: np.array([]))
    w_j_s: np.ndarray = field(default_factory=lambda: np.array([]))
    p_i_s_T: np.ndarray = field(default_factory=lambda: np.array([]))
    p_i_s_N: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_0: float = 1.0

    # ---- analytical demand / flow variables ----
    q_ijs: np.ndarray = field(default_factory=lambda: np.array([]))
    q_ijsb: np.ndarray = field(default_factory=lambda: np.array([]))
    q_ijsb_s_k: np.ndarray = field(default_factory=lambda: np.array([]))
    c_ijsb_s: np.ndarray = field(default_factory=lambda: np.array([]))
    d_ijsb_s_k: np.ndarray = field(default_factory=lambda: np.array([]))
    l_ijsb: np.ndarray = field(default_factory=lambda: np.array([]))
    f_h_ijsb: np.ndarray = field(default_factory=lambda: np.array([]))
    f_ijsb_s_k: np.ndarray = field(default_factory=lambda: np.array([]))
    H_prod_sb: np.ndarray = field(default_factory=lambda: np.array([]))
    L_prod_sb: np.ndarray = field(default_factory=lambda: np.array([]))
    H_j_p: np.ndarray = field(default_factory=lambda: np.array([]))
    H_nn_m: np.ndarray = field(default_factory=lambda: np.array([]))
    l_tilde_m_n: np.ndarray = field(default_factory=lambda: np.array([]))
    Y_j_p: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- analytical dual potentials ----
    xi_ijsb_k: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_m_nn_ijsb: np.ndarray = field(default_factory=lambda: np.array([]))
    mu_ijsb_n: np.ndarray = field(default_factory=lambda: np.array([]))
    mu_ijsb_s_k_n: np.ndarray = field(default_factory=lambda: np.array([]))
    nu_s_ij_n: np.ndarray = field(default_factory=lambda: np.array([]))
    mu_0js_n: np.ndarray = field(default_factory=lambda: np.array([]))

    def initialize_extended(self, params):
        S = params.n_sectors
        I = params.n_zones
        J = params.n_zones
        n_edges = params.network.n_edges
        n_nodes = params.network.n_nodes
        B = 1
        M = 1
        if self.h_ijsb.size == 0:
            self.h_ijsb = np.zeros((S, I, J, B))
        if self.l_bar_b.size == 0:
            self.l_bar_b = np.zeros((I, B))
        if self.l_tilde_b.size == 0:
            self.l_tilde_b = np.zeros((I, B))
        if self.q0_js.size == 0:
            self.q0_js = np.zeros((S, I))
        if self.h0_js.size == 0:
            self.h0_js = np.zeros((S, I))
        if self.Q_i0.size == 0:
            self.Q_i0 = np.zeros((S, I))
        if self.Q_0i.size == 0:
            self.Q_0i = np.zeros((S, I))
        if self.l_tilde_infra.size == 0:
            self.l_tilde_infra = np.zeros(I)
        if self.rho_js.size == 0:
            self.rho_js = np.zeros((S, I))
        if self.r_tilde_i.size == 0:
            self.r_tilde_i = np.zeros(I)
        if self.r_bar_b.size == 0:
            self.r_bar_b = np.zeros((I, B))
        if self.r_b_r.size == 0:
            self.r_b_r = np.zeros((I, B))
        if self.r_j_b_nr.size == 0:
            self.r_j_b_nr = np.zeros((I, B))
        if self.r_j_p.size == 0:
            self.r_j_p = np.zeros(I)
        if self.r_i_infra.size == 0:
            self.r_i_infra = np.zeros(I)
        if self.pi_j_s.size == 0:
            self.pi_j_s = np.zeros((S, I))
        if self.pi_j_p.size == 0:
            self.pi_j_p = np.zeros(I)
        if self.pi_nn_m.size == 0:
            self.pi_nn_m = np.zeros((M, n_edges))
        if self.w_j_s.size == 0:
            self.w_j_s = np.zeros((S, I))
        if self.w_m.size == 0:
            self.w_m = np.zeros(M)
        if self.p_i_s_T.size == 0:
            self.p_i_s_T = np.zeros((S, I))
        if self.p_i_s_N.size == 0:
            self.p_i_s_N = np.zeros((S, I))
        if self.y_k_P.size == 0:
            self.y_k_P = np.zeros(n_nodes)
        if self.Q_m_B.size == 0:
            self.Q_m_B = np.zeros((M, n_edges))
        if self.gamma_park.size == 0:
            self.gamma_park = np.zeros(n_nodes)
        if self.gamma_board.size == 0:
            self.gamma_board = np.zeros((M, n_edges))
        if self.V_m.size == 0:
            self.V_m = np.zeros((M, n_edges))
        if self.lambda_fleet_m.size == 0:
            self.lambda_fleet_m = np.zeros(M)
        if self.lambda_v_cons_mn.size == 0:
            self.lambda_v_cons_mn = np.zeros((M, n_nodes))
        if self.lambda_stat_mn.size == 0:
            self.lambda_stat_mn = np.zeros((M, n_nodes))
        if self.q_ijs.size == 0:
            self.q_ijs = np.zeros((S, I, J))
        if self.q_ijsb.size == 0:
            self.q_ijsb = np.zeros((S, I, J, B))
        if self.q_ijsb_s_k.size == 0:
            self.q_ijsb_s_k = np.zeros((S, I, J, S, B))
        if self.c_ijsb_s.size == 0:
            self.c_ijsb_s = np.zeros((S, S, I, J, B))
        if self.d_ijsb_s_k.size == 0:
            self.d_ijsb_s_k = np.zeros((S, S, I, J, B))
        if self.l_ijsb.size == 0:
            self.l_ijsb = np.zeros((S, I, J, B))
        if self.f_h_ijsb.size == 0:
            self.f_h_ijsb = np.zeros((S, I, J, B))
        if self.f_ijsb_s_k.size == 0:
            self.f_ijsb_s_k = np.zeros((S, I, J, S, B))
        if self.H_prod_sb.size == 0:
            self.H_prod_sb = np.zeros((S, I, B))
        if self.L_prod_sb.size == 0:
            self.L_prod_sb = np.zeros((S, I, B))
        if self.H_j_p.size == 0:
            self.H_j_p = np.zeros(I)
        if self.H_nn_m.size == 0:
            self.H_nn_m = np.zeros((M, n_edges))
        if self.l_tilde_m_n.size == 0:
            self.l_tilde_m_n = np.zeros((M, n_nodes))
        if self.Y_j_p.size == 0:
            self.Y_j_p = np.zeros(I)
        if self.xi_ijsb_k.size == 0:
            self.xi_ijsb_k = np.zeros((S, I, J, B, S))
        if self.chi_m_nn_ijsb.size == 0:
            self.chi_m_nn_ijsb = np.zeros((M, n_edges, S, I, J, B))
        if self.mu_ijsb_n.size == 0:
            self.mu_ijsb_n = np.zeros((S, I, J, B, n_nodes))
        if self.mu_ijsb_s_k_n.size == 0:
            self.mu_ijsb_s_k_n = np.zeros((S, I, J, S, B, n_nodes))
        if self.nu_s_ij_n.size == 0:
            self.nu_s_ij_n = np.zeros((S, I, J, n_nodes))
        if self.mu_0js_n.size == 0:
            self.mu_0js_n = np.zeros((S, J, n_nodes))


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
    alpha_Q: float = 0.2                          # volume update step for primal flow update
    alpha_Qhat: float = 0.3                       # legacy MSA damping parameter; actual outer-loop update follows theta_K = 1/(K+1)
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
    sa_T_init : float = 5.0 # initial temperature
    sa_alpha: float = 0.98  # cooling rate
    sa_noise_scale: float = 0.1 # scale of the noise added to the gradients

    # Inner-outer gradient step sizes
    alpha_Q: float = 0.05
    alpha_Qhat: float = 0.3
    alpha_lambda: float = 0.05
    eta_I: float = 0.01

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
    
