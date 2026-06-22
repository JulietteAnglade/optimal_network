import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_solver import *
from config import ExperimentConfig
from runner import solve_for_experiment



def make_grid_network_matrices(n_side=5, n_modes=1, default_infra=1.0, default_capacity=10.0):
    coords = []

    for i in range(n_side):
        for j in range(n_side):
            coords.append((i, j))

    n = len(coords)
    D = np.zeros((n, n))
    
    # Initialize infrastructure and capacity with zeros everywhere
    infra_single_mode = np.zeros((n, n))
    capacity_single_mode = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            x1, y1 = coords[i]
            x2, y2 = coords[j]

            # Manhattan distance
            manhattan_dist = abs(x1 - x2) + abs(y1 - y2)
            D[i, j] = manhattan_dist

            # Grid constraint: only immediate neighbors are connected
            if manhattan_dist == 1:
                infra_single_mode[i, j] = default_infra
                capacity_single_mode[i, j] = default_capacity

    # Replicate the grid structure across all modes
    # Shape: (n_modes, n_locations, n_locations)
    current_infra = np.repeat(infra_single_mode[np.newaxis, :, :], n_modes, axis=0)
    capacity = np.repeat(capacity_single_mode[np.newaxis, :, :], n_modes, axis=0)

    return D, current_infra, capacity





n_side = 5
n_locations = n_side**2
n_modes = 1
n_sectors = 1

distance_matrix, current_infra_level, capacity = make_grid_network_matrices(n_side, n_modes, default_infra = 10, default_capacity=100)
print(distance_matrix, current_infra_level, capacity)

# Utility function

alpha_c=np.array([[0.6]])          # (S,K)
alpha_l=np.array([0.2])          # (S,)
alpha_f=np.array([
    0.2
])          # (S,)
utility=CobbDouglasUtility(
    alpha_c=alpha_c,
    alpha_l=alpha_l,
    alpha_f=alpha_f
)

# Production function
A=np.ones(
    (n_sectors,n_locations)
)

aH=np.array([
    0.3
])

aL=np.array([
    0.3
])

aQ=np.array([
    [0.2]
])

# total = .3+.3+.2=.8 <1
# important pour invert()

production=CobbDouglasProduction(
    A=A,
    aH=aH,
    aL=aL,
    aQ=aQ
)



base_config = ExperimentConfig(

    name="debug",


    n_locations=n_locations,
    n_modes=n_modes,
    types_of_modes = [ModeOfTransport(name='Car', speed=50.0, cost_per_km=1.2)],

    distance_matrix=distance_matrix,

    current_infra_level=current_infra_level,

    capacity=capacity,

    constraints=[[1,1]],

    
    n_sectors=n_sectors,

    production_function= production,
    utility_function=utility,

    L_bar=30,
    H_bar=1,
    q_bar=1000,
    K=1000,

    # congestion OFF
    congestion_a=0.5,
    congestion_b=1.0,

    tau_link= IcebergTau,
    t_link= BPRTime,

    sector_dependance_tau=np.array([1.0]),


    d_w=1,
    phi=np.array([0.5]),
    sigma=np.array([5]),
    sigma_tilde=3,

    # algo (petites valeurs pour debug)
    T_outer=100,
    eta_p=0.001,
    eta_w=0.001,
    eta_r=0.001,

    T_allocation=10000,

    eta_beta=0.01,
    eta_I=0.01,
    eta_H=0.01,

    T_inner=1000,

    alpha_Q=0.0001,
    alpha_Qhat=0.0001,
    eta_gamma=0.0001,

    tol_outer=1e-4,
    tol_inner=1e-4,

    verbose=True,
    log_every=50,

    decay_start=0,
    decay_type=None
)

state, params = solve_for_experiment(base_config)

print("Done")

print("Population:")
print(state.L)

print("Wages:")
print(state.w)

print("Prices:")
print(state.p)

print("\nDiagnostics")

print(
    "NaN population:",
    np.isnan(state.L).any()
)

print(
    "NaN wages:",
    np.isnan(state.w).any()
)

print(
    "Population sum:",
    state.L.sum()
)

print(
    "Target population:",
    params.L_bar
)