"""Toy end-to-end run of model_solver_v2: 3 zones, 2 sectors (one
tradable, one non-tradable activity purpose), 1 residential / 1
non-residential / 1 parking building type, 1 transit mode, on a small
6-car-edge + 4-transit-edge + 2-boundary-edge network.

Run with: python -m model_solver_v2.demo.run_demo
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_solver_v2.algo_params import AlgoParams
from model_solver_v2.functional_forms import (
    BPRBoardingTime, BPRCongestionTime, CobbDouglasActivityUtilityN,
    CobbDouglasActivityUtilityT, CobbDouglasCommercialProduction,
    CobbDouglasParkingProduction, CobbDouglasTransitVehicleProduction,
    CobbDouglasUtility, ExponentialIcebergTau, LogStationCapacity,
    PowerParkingSearchTime, PowerWaitingTime,
)
from model_solver_v2.network import NetworkBuilder
from model_solver_v2.params import ModelParams
from model_solver_v2.solver import GeneralizedSpatialEquilibriumSolver
from model_solver_v2.state import ModelState


def build_network():
    n_zones, n_modes = 3, 1
    b = NetworkBuilder(n_zones=n_zones, n_modes_transit=n_modes)
    car_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
    for u, v in car_pairs:
        b.add_car_edge(u, v, kappa=1.0, I_min=0.5, I_max=5.0)
    transit_pairs = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for u, v in transit_pairs:
        b.add_transit_edge(u, v, mode=0, kappa=2.0, I_min=0.5, I_max=5.0)
    # boundary connections (entry_node = n_zones = 3) to gateway zone 0
    b.add_car_edge(b.entry_node, 0, kappa=1.0, I_min=0.5, I_max=5.0)
    b.add_car_edge(0, b.entry_node, kappa=1.0, I_min=0.5, I_max=5.0)
    return b.build()


def build_params(network):
    S, I, A = 2, 3, 2
    Br, Bnr, P, M = 1, 1, 1, 1
    is_tradable = np.array([True, False])
    n_trad = int(is_tradable.sum())

    utility = CobbDouglasUtility(alpha_c=np.full(n_trad, 0.3), alpha_l=0.3, alpha_fh=0.4)
    activity_utility_T = CobbDouglasActivityUtilityT(alpha_d=np.full(n_trad, 0.5), alpha_f=np.full(n_trad, 0.5))
    activity_utility_N = CobbDouglasActivityUtilityN(alpha_f=np.full(S - n_trad, 1.0))
    commercial_production = CobbDouglasCommercialProduction(
        A=np.ones((S, I)), aH=np.full(S, 0.3), aL=np.full((S, Bnr), 0.2), aQ=np.full((S, n_trad), 0.1))
    parking_production = CobbDouglasParkingProduction(A=np.ones((I, P)), aH=np.full(P, 0.5), aL=np.full(P, 0.4))
    transit_vehicle_production = CobbDouglasTransitVehicleProduction(A=np.ones((M, network.n_edges)), a=np.full(M, 0.5))
    station_capacity = LogStationCapacity(A=np.full(M, 1.0))
    congestion_time = BPRCongestionTime(T0=np.full(network.n_edges, 1.0), a=0.6, b=4.0)
    iceberg_tau = ExponentialIcebergTau(
        delta=np.full(S, 0.05), mask=network.freight_allowed.astype(float),
        tau_scale=np.ones(network.n_edges), a=0.6, b=4.0)
    parking_search_time = PowerParkingSearchTime(c0=np.full(I, 0.1), b=1.0)
    boarding_time = BPRBoardingTime(T0=np.full((M, network.n_edges), 0.1), a=0.6, b=2.0)
    waiting_time = PowerWaitingTime(c0=np.full((M, network.n_edges), 0.5), b=1.0)

    return ModelParams(
        n_zones=I, n_sectors=S, n_activities=A, is_tradable=is_tradable,
        n_b_r=Br, n_b_nr=Bnr, n_p=P, n_modes_transit=M,
        network=network,
        utility=utility, activity_utility_T=activity_utility_T, activity_utility_N=activity_utility_N,
        commercial_production=commercial_production, parking_production=parking_production,
        transit_vehicle_production=transit_vehicle_production, station_capacity=station_capacity,
        congestion_time=congestion_time, iceberg_tau=iceberg_tau,
        parking_search_time=parking_search_time, boarding_time=boarding_time, waiting_time=waiting_time,
        H_bar=10.0, L_bar=np.full(I, 100.0), q_bar=300.0, q_bar_0=50.0, K=1000.0,
        is_gateway=np.array([True, False, False]),
        wage_parking=np.full(I, 1.0),
        V_bar_m=np.full(M, 50.0), H_allocated_m=np.full(M, 20.0),
        omega_w=1.0, omega_sprime=np.full(A, 0.5),
        sigma_sprime=np.full(A, 1.0), sigma_loc=np.full(S, 1.0), sigma_s=np.full(S, 1.0), sigma_tilde=1.0,
        phi=np.full(S, 1.0), tau_E=np.full(S, 1.0), tau_M=np.full(S, 1.0),
        kappa_floor_res=np.full((I, Br), 1.0), kappa_floor_nonres=np.full((I, Bnr), 1.0),
        kappa_floor_park=np.full((I, P), 1.0),
        alpha_floor_res=np.full((I, Br), 0.5), alpha_floor_nonres=np.full((I, Bnr), 0.5),
        alpha_floor_park=np.full((I, P), 0.5),
        alpha_infra=0.5,
    )


def main():
    network = build_network()
    params = build_params(network)
    state = ModelState.zeros(params)
    # Conservative step sizes: this toy calibration (arbitrary Cobb-Douglas
    # shares, unit costs, etc.) is not fit to real data, and several
    # primitives involve reciprocal terms (parking search time ~ 1/S,
    # Cobb-Douglas inverse demand ~ 1/price); small steps keep the demo
    # inside the well-behaved region long enough to show every block
    # firing. Reaching genuine multi-cycle convergence on a specific
    # scenario is a calibration exercise, not an algorithmic one.
    algo = AlgoParams(T_inner=8, T_outer=4, alpha_x=0.002, alpha_lambda=0.002,
                      alpha_l=0.01, alpha_Y=0.01, alpha_V=0.01, alpha_I=0.01, alpha_beta=0.01,
                      routing_mode="discrete")

    solver = GeneralizedSpatialEquilibriumSolver()
    solver.solve(state, params, algo)

    print("Outer-loop history:")
    for entry in solver.history:
        print(f"  K={entry['K']:2d}  residual={entry['residual']:.6f}  inner_iters={entry['inner_iters']}")

    print("\nFinal state summary:")
    print(f"  Q_hat_nn (edge flows): {np.round(state.Qhat_nn, 3)}")
    print(f"  I_nn (infrastructure): {np.round(state.I_nn, 3)}")
    print(f"  beta (capital shadow price): {state.beta:.4f}")
    print(f"  q_bar_s (population by sector): {np.round(state.q_bar_s, 3)}")
    print(f"  Y_js (sector output): {np.round(state.Y_js, 3)}")


if __name__ == "__main__":
    main()
