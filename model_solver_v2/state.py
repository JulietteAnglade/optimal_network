"""Model state: every variable in algorithm.tex Tables 1 & 2, plus the
outer macroscopic-coupling vector Z, the outer structural primal O, and
their associated routing potentials / duals.

Shape conventions (S = n_sectors, I = n_zones, Br/Bnr/P = n_b_r/n_b_nr/n_p,
A = n_activities, M = n_modes_transit, E = n_edges, Nn = n_nodes):

  HG  = (S, I, I, Br)        "household group" (s, i, j, b), b in B^r
  HGA = HG + (A,)            + activity purpose s' in S^P
  HGAK = HG + (I, A)         + activity destination zone k, purpose s'
         (k precedes s' so that s' -- the axis the primitive functions in
         functional_forms.py broadcast their per-activity parameters
         against -- is always the trailing axis)

Every attribute name below is the ASCII transliteration of its LaTeX
symbol; the comment gives the symbol, its shape, and the defining
equation number(s) so the array can be audited directly against the
document. Dense allocation of the (s',k)-indexed secondary-trip arrays
is a demo-scale simplification -- Section 5.2 of the document itself
notes that production-scale networks require CSR/sparse storage for
these objects.
"""
from dataclasses import dataclass, field

import numpy as np

from .params import ModelParams


@dataclass
class ModelState:
    # =====================================================================
    # Iterative Primal X (Table 1; updated by eq. 32-41)
    # =====================================================================
    h_ijsb: np.ndarray          # h^{ijsb}                HG          eq.32
    l_tilde_res: np.ndarray     # tilde l_i^b, b in B^r    (I, Br)     eq.33
    l_tilde_nonres: np.ndarray  # tilde l_i^b, b in B^nr   (I, Bnr)    eq.33
    l_tilde_park: np.ndarray    # tilde l_i^b, b in P      (I, P)      eq.33
    Y_js: np.ndarray            # Y_j^s                    (S, I)      eq.34
    Q_ij_s: np.ndarray          # Q_{ij}^s                 (S, I, I)   eq.35
    Qtilde_ij_s: np.ndarray     # tilde Q_{ij}^s           (S, I, I)   eq.36
    q0_js: np.ndarray           # q^{0,js}                 (S, I)      eq.37
    h0_js: np.ndarray           # h^{0,js}                 (S, I)      eq.38
    Q_i0: np.ndarray            # Q_{i0}^s                 (S, I)      eq.39
    Q_0i: np.ndarray            # Q_{0i}^s                 (S, I)      eq.40
    l_tilde_infra: np.ndarray   # tilde l_i^{infra}        (I,)        eq.41

    # =====================================================================
    # Iterative Dual Lambda (Table 1; updated by eq. 42-60)
    # =====================================================================
    lambda0: float              # lambda^0                 scalar      eq.42
    w0: float                   # w^0                      scalar      eq.43
    rho_js: np.ndarray          # rho_j^s                  (S, I)      eq.44
    r_tilde: np.ndarray         # tilde r_i                (I,)        eq.45
    r_bar_res: np.ndarray       # bar r_i^b, b in B^r       (I, Br)     eq.46
    r_bar_nonres: np.ndarray    # bar r_i^b, b in B^nr      (I, Bnr)    eq.46
    r_bar_park: np.ndarray      # bar r_i^b, b in P         (I, P)      eq.46
    r_res: np.ndarray           # r_i^{b,r}                (I, Br)     eq.47
    r_nonres: np.ndarray        # r_j^{b,nr}               (I, Bnr)    eq.48
    r_park: np.ndarray          # r_j^p                    (I, P)      eq.49
    r_infra: np.ndarray         # r_i^{infra}              (I,)        eq.50
    pi_js: np.ndarray           # pi_j^s                   (S, I)      eq.51
    pi_jp: np.ndarray           # pi_j^p                   (I, P)      eq.52
    pi_nn_m: np.ndarray         # pi_{nn'}^m               (M, E)      eq.53
    w_js: np.ndarray            # w_j^s                    (S, I)      eq.54
    w_m: np.ndarray             # w_m                      (M,)        eq.55
    p_sT: np.ndarray            # p_i^{s',T}               (S, I)      eq.56
    p_sN: np.ndarray            # p_i^{s',N}               (S, I)      eq.58
    theta0: float                # theta_0                  scalar      eq.59
    lambda_stat_mn: np.ndarray  # lambda^{stat}_{m,n}      (M, Nn)     eq.60

    # =====================================================================
    # Analytical Primal Y (Table 1; computed by eq. 23-31)
    # =====================================================================
    q_bar_s: np.ndarray         # bar q^s                  (S,)        eq.19
    q_ijs: np.ndarray           # q^{ijs}                  (S, I, I)   eq.20
    q_ijsb: np.ndarray          # q^{ijsb}                 HG          eq.21
    q_ijsb_sk: np.ndarray       # q^{ijsb}_{s'k}           HGAK        eq.22
    c_ijsb_s: np.ndarray        # c_{s'}^{ijsb}            HGA (s' in S^T only, else 0)  eq.23
    d_ijsb_sk: np.ndarray       # d_{s'k}^{ijsb}           HGAK (s' in S^T only)         eq.24
    l_ijsb: np.ndarray          # l^{ijsb}                 HG          eq.25
    fh_ijsb: np.ndarray         # f_h^{ijsb}               HG          eq.26
    f_ijsb_sk: np.ndarray       # f_{s'k}^{ijsb}           HGAK        eq.27
    H_js: np.ndarray            # H_j^s                    (S, I)      eq.29
    L_jsb: np.ndarray           # L_j^{sb}, b in B^nr      (S, I, Bnr) eq.29
    Q_jss: np.ndarray           # Q_j^{ss'}, s' in S^T     (S, I, A)   eq.30
    H_jp: np.ndarray            # H_j^p                    (I, P)      eq.31
    H_nn_m: np.ndarray          # H_{nn'}^m                (M, E)      eq.31
    l_tilde_mn: np.ndarray      # tilde l^m_n              (M, Nn)     eq.31

    # =====================================================================
    # Analytical Dual K -- routing potentials (Section 3, eq. 5-13)
    # =====================================================================
    xi_ijsb_k: np.ndarray       # xi_k^{ijsb}              HG + (I,)        eq.28
    chi_m_nn_ijsb: np.ndarray   # chi_{m,nn',ijsb}         (M, E) + HG      eq.6
    mu_ijsb_n: np.ndarray       # mu_n^{ijsb}              HG + (Nn,)       eq.5
    mu_ijsb_sk_n: np.ndarray    # mu_{s'k,n}^{ijsb}        HGAK + (Nn,)     eq.7
    nu_s_ij_n: np.ndarray       # nu_{ij,n}^s              (S, I, I, Nn)    eq.11-13
    mu_0js_n: np.ndarray        # mu_n^{0js}               (S, I, Nn)       eq.8

    # =====================================================================
    # Outer macroscopic coupling Z (MSA, eq. 61)
    # =====================================================================
    Qhat_nn: np.ndarray         # hat Q_{nn'}              (E,)
    yP_k: np.ndarray            # y_k^P                    (I,)
    Q_mB_nn: np.ndarray         # Q_{nn'}^{m,B}            (M, E)
    gamma_nn: np.ndarray        # gamma_{nn'}              (E,)        eq.71
    gamma_park_k: np.ndarray    # gamma_k^{park}           (I,)        eq.73
    gamma_board_m_nn: np.ndarray  # gamma_{m,nn'}^{board}  (M, E)      eq.72

    # =====================================================================
    # Outer structural primal O union {beta} (Table 2, eq. 62-74)
    # =====================================================================
    I_nn: np.ndarray            # I_{nn'}                  (E,)        eq.62-63
    l_bar_res: np.ndarray       # bar l_i^b, b in B^r       (I, Br)     eq.64
    l_bar_nonres: np.ndarray    # bar l_i^b, b in B^nr      (I, Bnr)    eq.64
    l_bar_park: np.ndarray      # bar l_i^b, b in P         (I, P)      eq.64
    Y_pk: np.ndarray            # Y_k^p                    (I, P)      eq.65-66
    V_m_nn: np.ndarray          # V_{nn'}^m                (M, E)      eq.68
    lambda_fleet_m: np.ndarray  # lambda^{fleet}_m         (M,)        eq.69
    lambda_vcons_mn: np.ndarray  # lambda^{v_cons}_{m,n}   (M, Nn)     eq.70
    beta: float                  # beta                     scalar      eq.74

    # ---- simulated-annealing / multi-start bookkeeping for I_{nn'} ----
    sa_temperature: float = 1.0
    sa_best_I: np.ndarray = field(default_factory=lambda: np.array([]))
    sa_best_objective: float = -np.inf

    # =====================================================================
    # Derived routing caches (not individually named in the document, but
    # required to evaluate eq. 62/71's flow-aggregation terms and eq. 32's
    # car-mode-transition imbalance; populated by routing.py each inner
    # iteration, consumed by inner_loop.py / outer_loop.py).
    # =====================================================================
    flow_time_weighted_nn: np.ndarray = field(default_factory=lambda: np.array([]))
    # (E,) := sum_{ijsb} rho_j^s * omega_w * (q_{nn'}^{ijsb} + sum_{s'k} q_{s'k,nn'}^{ijsb})

    flow_raw_nn: np.ndarray = field(default_factory=lambda: np.array([]))
    # (E,) := sum_{ijsb} (q_{nn'}^{ijsb} + sum_{s'k} q_{s'k,nn'}^{ijsb}) + sum_{js} q^{0,js}_{nn'}
    # (un-weighted physical flow, used to form Q_hat^{induced} for the MSA step, eq.61)

    flow_goods_nn_s: np.ndarray = field(default_factory=lambda: np.array([]))
    # (S, E) := sum_{ij} Q_{ij,nn'}^s  (aggregated freight edge flow per sector)

    flow_goods_value_nn_s: np.ndarray = field(default_factory=lambda: np.array([]))
    # (S, E) := sum_{ij} nu_{ij,n'}^s * Q_{ij,nn'}^s  (price-weighted freight edge flow)

    car_flow_imbalance_ijsb_k: np.ndarray = field(default_factory=lambda: np.array([]))
    # HG + (I,) := sum_{nk in L^car} q_{nk}^{ijsb,t} - sum_{kn in L^car} q_{kn}^{ijsb,t}, eq.32

    @staticmethod
    def zeros(params: ModelParams) -> "ModelState":
        S, I = params.n_sectors, params.n_zones
        Br, Bnr, P = params.n_b_r, params.n_b_nr, params.n_p
        A, M = params.n_activities, params.n_modes_transit
        net = params.network
        E, Nn = net.n_edges, net.n_nodes
        HG = (S, I, I, Br)
        HGA = HG + (A,)
        HGAK = HG + (I, A)

        return ModelState(
            h_ijsb=np.zeros(HG),
            l_tilde_res=np.zeros((I, Br)),
            l_tilde_nonres=np.zeros((I, Bnr)),
            l_tilde_park=np.zeros((I, P)),
            # Anchored to the aggregate-demand scale (q_bar/n_sectors) rather
            # than an arbitrary constant: eq. 29-30 makes H_j^s/L_j^{sb}/Q_j^{ss'}
            # proportional to Y_j^s (the "given production target" shortcut),
            # and with decreasing-returns Cobb-Douglas (sum of elasticities < 1)
            # F(H,L,Q) scales as Y_j^s^{<1}, which *exceeds* Y_j^s whenever
            # Y_j^s < 1 -- pushing pi_j^s (eq. 51) down and Y_j^s (eq. 34) down
            # further, an inescapable runaway collapse once Y_j^s dips under 1.
            # Starting near the population's actual consumption scale avoids
            # ever entering that regime; erring larger is safe (oversupply
            # just lowers price, no analogous multiplicative trap), so no
            # upper guard is needed, only a floor for pathologically small q_bar.
            Y_js=np.full((S, I), max(params.q_bar / S, 1.0)),
            Q_ij_s=np.zeros((S, I, I)),
            Qtilde_ij_s=np.zeros((S, I, I)),
            q0_js=np.zeros((S, I)),
            h0_js=np.zeros((S, I)),
            Q_i0=np.zeros((S, I)),
            Q_0i=np.zeros((S, I)),
            l_tilde_infra=np.zeros(I),

            lambda0=0.0,
            w0=0.0,
            rho_js=np.full((S, I), 1.0),
            r_tilde=np.full(I, 1.0),
            r_bar_res=np.full((I, Br), 1.0),
            r_bar_nonres=np.full((I, Bnr), 1.0),
            r_bar_park=np.full((I, P), 1.0),
            r_res=np.full((I, Br), 1.0),
            r_nonres=np.full((I, Bnr), 1.0),
            r_park=np.full((I, P), 1.0),
            r_infra=np.full(I, 1.0),
            pi_js=np.full((S, I), 1.0),
            pi_jp=np.full((I, P), 1.0),
            pi_nn_m=np.full((M, E), 1.0),
            w_js=np.full((S, I), 1.0),
            w_m=np.full(M, 1.0),
            p_sT=np.full((S, I), 1.0),
            p_sN=np.full((S, I), 1.0),
            theta0=1.0,
            lambda_stat_mn=np.zeros((M, Nn)),

            q_bar_s=np.full(S, params.q_bar / S),
            q_ijs=np.full((S, I, I), params.q_bar / (S * I * I)),
            q_ijsb=np.full(HG, params.q_bar / (S * I * I * Br)),
            q_ijsb_sk=np.zeros(HGAK),
            c_ijsb_s=np.zeros(HGA),
            d_ijsb_sk=np.zeros(HGAK),
            l_ijsb=np.full(HG, 1.0),
            fh_ijsb=np.full(HG, 1.0),
            f_ijsb_sk=np.full(HGAK, 1.0),
            H_js=np.full((S, I), 1.0),
            L_jsb=np.full((S, I, Bnr), 1.0),
            Q_jss=np.zeros((S, I, A)),
            H_jp=np.full((I, P), 1.0),
            H_nn_m=np.full((M, E), 1.0),
            l_tilde_mn=np.full((M, Nn), 1.0),

            xi_ijsb_k=np.zeros(HG + (I,)),
            chi_m_nn_ijsb=np.zeros((M, E) + HG),
            mu_ijsb_n=np.zeros(HG + (Nn,)),
            mu_ijsb_sk_n=np.zeros(HGAK + (Nn,)),
            nu_s_ij_n=np.zeros((S, I, I, Nn)),
            mu_0js_n=np.zeros((S, I, Nn)),

            Qhat_nn=np.full(E, 0.1),
            yP_k=np.zeros(I),
            Q_mB_nn=np.zeros((M, E)),
            gamma_nn=np.zeros(E),
            gamma_park_k=np.zeros(I),
            gamma_board_m_nn=np.zeros((M, E)),

            I_nn=np.maximum(net.I_min, 1.0),
            l_bar_res=np.full((I, Br), 1.0),
            l_bar_nonres=np.full((I, Bnr), 1.0),
            l_bar_park=np.full((I, P), 1.0),
            Y_pk=np.full((I, P), 1.0),
            V_m_nn=np.full((M, E), 1.0),
            lambda_fleet_m=np.zeros(M),
            lambda_vcons_mn=np.zeros((M, Nn)),
            beta=1.0,

            sa_temperature=1.0,
            sa_best_I=np.maximum(net.I_min, 1.0).copy(),
            sa_best_objective=-np.inf,

            flow_time_weighted_nn=np.zeros(E),
            flow_raw_nn=np.zeros(E),
            flow_goods_nn_s=np.zeros((S, E)),
            flow_goods_value_nn_s=np.zeros((S, E)),
            car_flow_imbalance_ijsb_k=np.zeros(HG + (I,)),
        )
