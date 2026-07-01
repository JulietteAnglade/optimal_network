"""Model parameters: index-set sizes, primitive-function instances, and
economic constants (algorithm.tex throughout; constants are collected
here rather than re-declared at every site they appear).

Index-set conventions adopted where the document reuses notation across
sections without re-declaring the index set explicitly (documented here
so they are auditable against the LaTeX):

- Zones: a single index set `I` is used for home / work / destination
  (i, j, k), size `n_zones`.
- Sectors `S` (== `S^GS`, commercial production sectors): size
  `n_sectors`. Indexes `Y_j^s`, freight `Q_ij^s`, prices `p^{s,T}`, `p^{s,N}`.
- Activity purposes `S^P = S^T cup S^N`: a single dimension `n_activities`
  with a boolean mask `is_tradable` marking `S^T`. This same set indexes
  at-home shipped consumption `c_{s'}`, activity demand `d_{s'k}`, and the
  intermediate-input index `s'` of `Q_j^{s s'}` (eq. 30) -- the document
  itself reuses `s' in S^T` identically across SS1.2, 4, 5.1.
- Building types are kept as three separate families `B^r`, `B^nr`,
  `P -= S^park`, each its own array dimension (`n_b_r`, `n_b_nr`, `n_p`),
  since every equation already disambiguates by superscript.
- Transit modes `M^transit`: size `n_modes_transit`.
"""
from dataclasses import dataclass

import numpy as np

from .network import Network
from .primitives import (
    UtilityPrimitive, ActivitySubUtilityT, ActivitySubUtilityN,
    CommercialProductionPrimitive, ParkingProductionPrimitive,
    TransitVehicleProductionPrimitive, StationCapacityPrimitive,
    CongestionTime, IcebergTau, ParkingSearchTime, BoardingTime, WaitingTime,
)


@dataclass
class ModelParams:
    # ---- Sizes ----
    n_zones: int            # |I| = |J|
    n_sectors: int          # |S| = |S^GS|
    n_activities: int       # |S^P|
    is_tradable: np.ndarray # (n_activities,) bool mask of S^T subset of S^P
    n_b_r: int              # |B^r|
    n_b_nr: int             # |B^nr|
    n_p: int                # |P| = |S^park|
    n_modes_transit: int    # |M^transit|

    network: Network

    # ---- Primitive functions (Section 1) ----
    utility: UtilityPrimitive                                  # U(c, l, f_h)
    activity_utility_T: ActivitySubUtilityT                    # u_{s'}(d, f), s' in S^T
    activity_utility_N: ActivitySubUtilityN                    # u_{s'}(f), s' in S^N
    commercial_production: CommercialProductionPrimitive       # F_j^s
    parking_production: ParkingProductionPrimitive             # F_j^p
    transit_vehicle_production: TransitVehicleProductionPrimitive  # F^m_{nn'}
    station_capacity: StationCapacityPrimitive                 # F^m_n
    congestion_time: CongestionTime                            # T_{nn'}
    iceberg_tau: IcebergTau                                    # tau^s_{nn'}
    parking_search_time: ParkingSearchTime                     # T^P_k
    boarding_time: BoardingTime                                # T^{m,board}_{nn'}
    waiting_time: WaitingTime                                  # t^{m,W}_{nn'}

    # ---- Endowments / economy-wide constants ----
    H_bar: float            # overline{H}, total time per capita
    L_bar: np.ndarray       # (n_zones,) land endowment per zone
    q_bar: float            # bar q, total population
    q_bar_0: float          # bar q^0, peripheral labor pool capacity
    K: float                # capital budget
    is_gateway: np.ndarray  # (n_zones,) bool, i in N^G: the zone(s) economically
                             # identified with the boundary node n_g, used by the
                             # indicator terms in eq. 37/39/40/56/57/59. Kept
                             # separate from `network.entry_node` (the *graph*
                             # node peripheral routing distances are computed
                             # to/from): the document indexes N^G as a node set
                             # but eq. 57 tests membership of a *zone* i in it,
                             # so a zone-level flag is the well-defined object.
    wage_parking: np.ndarray  # (n_zones,) w_j^p: the document references this
                               # symbol in eq. 31/52 but never defines a market-
                               # clearing equation for it (Table 1 tracks only
                               # w_j^s and w_m as dual wage variables) -- treated
                               # here as an exogenous parking-sector wage.
    V_bar_m: np.ndarray     # (M,) fleet-size cap per transit mode
    H_allocated_m: np.ndarray  # (M,) labor allocated to transit operations per mode

    # ---- Weights / monetization ----
    omega_w: float          # time -> cost (monetary) conversion factor
    omega_sprime: np.ndarray  # (n_activities,) weight of activity purpose s' in cascade

    # ---- Nested-logit dispersion parameters ----
    sigma_sprime: np.ndarray  # (n_activities,) activity-destination nest, eq. 15
    sigma_loc: np.ndarray     # (n_sectors,) residence/building-choice nest, eq. 17 (sigma_{ijs})
    sigma_s: np.ndarray       # (n_sectors,) OD nest, eq. 18
    sigma_tilde: float        # top-level sector-choice nest, eq. 19

    # ---- Trade / freight ----
    phi: np.ndarray         # (n_sectors,) weight of good s per unit volume
    tau_E: np.ndarray       # (n_sectors,) export iceberg-type factor at the boundary
    tau_M: np.ndarray       # (n_sectors,) import iceberg-type factor at the boundary

    # ---- Capital-budget unit costs / depreciation ----
    kappa_floor_res: np.ndarray     # (n_zones, n_b_r)  bar_kappa_{i,b}, b in B^r
    kappa_floor_nonres: np.ndarray  # (n_zones, n_b_nr) bar_kappa_{i,b}, b in B^nr
    kappa_floor_park: np.ndarray    # (n_zones, n_p)    bar_kappa_{i,b}, b in P
    alpha_floor_res: np.ndarray     # (n_zones, n_b_r)  alpha_i^b
    alpha_floor_nonres: np.ndarray  # (n_zones, n_b_nr) alpha_i^b
    alpha_floor_park: np.ndarray    # (n_zones, n_p)    alpha_i^b
    alpha_infra: float              # alpha^infra

    def __post_init__(self):
        # The document indexes activity-purpose prices and outputs by s'
        # directly into the production-sector arrays (p_k^{s',T}/p_k^{s',N}
        # in eq. 14, Q_j^{ss'} in eq. 30, Y_i^{s'} in eq. 56/58): S^P is the
        # *same* index set as S, partitioned into S^T/S^N by `is_tradable`,
        # not an independently-sized set. This identification is what lets
        # activity-good prices reuse `ModelState.p_sT` / `p_sN` directly.
        if self.n_activities != self.n_sectors:
            raise ValueError(
                "n_activities must equal n_sectors: activity purposes S^P "
                "are the production-sector set S, partitioned into S^T/S^N "
                "by `is_tradable` (see eq. 14, 30, 56, 58)."
            )
