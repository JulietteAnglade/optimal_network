import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run import run
from model_solver.network_encoder import ModeOfTransport
from model_solver.functions import CobbDouglasUtility, CobbDouglasProduction, IcebergTau, BPRTime


def main():
    # --------------------- 1. real world network ---------------------
    n_locations = 4
    n_modes = 2

    #Interpretation:
    #0 financial center
    #1 manufacturing
    #2 suburban residential area
    #3 logistics area

    types_of_modes = [
        ModeOfTransport(name='Car', speed=55.0, cost_per_km=1.2),
        ModeOfTransport(name='Train', speed=110.0, cost_per_km=0.45),
    ]

    distance_matrix = np.array([
        [0.0, 18, 30, 40],
        [18, 0.0, 15, 22],
        [30, 15, 0, 20], 
        [40, 22, 20, 0]
    ])
    # Shape: (n_locations, n_locations, n_modes) = (4, 4, 2)
    """ current_infra_level = np.array([
        # location 0
        [[3.0, 4.0], [2.0, 3.5], [1.5, 1.0], [1.0, 2.5]],
        # location 1
        [[2.0, 3.5], [3.0, 4.0], [2.5, 1.5], [2.0, 3.0]],
        # location 2
        [[1.5, 1.0], [2.5, 1.5], [3.0, 2.0], [1.8, 1.0]],
        # location 3
        [[1.0, 2.5], [2.0, 3.0], [1.8, 1.0], [3.0, 3.5]],
    ]) """

    current_infra_level = np.array([
        # location 0
        [[4.0, 8.0], [1.80, 7.0], [1.0, 1.50], [8.0, 4.0]],
        # location 1
        [[1.80, 7.0], [4.0, 8.0], [2.0, 2.0], [1.0, 5.0]],
        # location 2
        [[1.0, 1.0], [2.0, 2.0], [3.0, 2.0], [1.0, 1.0]],
        # location 3
        [[8.0, 4.0], [1.0, 5.0], [1.0, 1.0], [3.0, 7.0]],
    ])

    # Shape: (n_locations, n_locations, n_modes) = (4, 4, 2)
    capacity = np.array([
        # location 0
        [[40.0, 80.0], [18.0, 70.0], [10.0, 15.0], [8.0, 45.0]],
        # location 1
        [[18.0, 70.0], [40.0, 80.0], [20.0, 20.0], [16.0, 55.0]],
        # location 2
        [[10.0, 15.0], [20.0, 20.0], [35.0, 25.0], [14.0, 12.0]],
        # location 3
        [[8.0, 45.0], [16.0, 55.0], [14.0, 12.0], [35.0, 70.0]],
    ])

    constraints = np.array([
        [1, 1],  # goods_allowed, commuters_allowed for car
        [0, 1],  # goods_allowed, commuters_allowed for train
    ], dtype=bool)

    # ----------------------- 2. model parameters -----------------------
    n_sectors = 3
    #0 advanced services
    #1 manufacturing
    #2 logistics

    A=np.array([
        # services
        [1.60, 1.00, 0.85, 0.95],
        # manufacturing
        [0.90, 1.55, 0.95, 1.10],
        # logistics
        [0.80, 1.10, 0.90, 1.65],
    ])

    # labor elasticities
    aH=np.array([0.55,0.50,0.48])
    aL = np.array([0.15, 0.18, 0.2])
    aQ = np.array([
    [0.00, 0.10, 0.18],
    [0.12, 0.00, 0.16],
    [0.10, 0.12, 0.00],
    ])

    production_function = CobbDouglasProduction(A, aH, aL, aQ)

    utility_function = CobbDouglasUtility(
    alpha_c=np.array([
        [0.45, 0.20, 0.15],
        [0.20, 0.45, 0.15],
        [0.25, 0.25, 0.30],
    ]),
    alpha_l=np.array([0.12, 0.12, 0.12]),
    alpha_f=np.array([0.08, 0.08, 0.08]),
)

    L_bar = np.array([6.0, 10.0, 18.0, 12.0])
    H_bar = 1.0
    q_bar = 180.0
    K = 250

    sector_dependance_tau = np.array([0.01, 0.03, 0.05])
    tau_link = IcebergTau
    t_link = BPRTime

    d_w = 1.0
    phi = np.array([0.8, 1.8, 2.5])
    sigma = np.array([0.45, 0.50, 0.60])
    sigma_tilde = 0.75

    # ----------------- 3. algo hyperparameters ------------------------
    T_outer = 2000
    eta_p = 0.01
    eta_w = 0.01
    eta_r = 0.01
    T_allocation = 500
    eta_beta = 0.025
    eta_I = 0.025
    eta_H = 0.01
    T_inner = 100
    alpha_Q = 0.01
    alpha_Qhat = 0.01
    eta_gamma = 0.01
    tol_outer = 5e-3
    tol_inner = 1e-4
    verbose = True
    log_every = 50
    decay_start = 0
    decay_type = None


    example_name = 'test_example2_BIS'

    # ----------------- 4. run ----------------------------------------
    
    run(
        n_locations,
        n_modes,
        types_of_modes,
        distance_matrix,
        current_infra_level,
        capacity,
        constraints,
        n_sectors,
        production_function,
        utility_function,
        L_bar,
        H_bar,
        q_bar,
        K,
        sector_dependance_tau,
        tau_link,
        t_link,
        d_w,
        phi,
        sigma,
        sigma_tilde,
        T_outer,
        eta_p,
        eta_w,
        eta_r,
        T_allocation,
        eta_beta,
        eta_I,
        eta_H,
        T_inner,
        alpha_Q,
        alpha_Qhat,
        eta_gamma,
        tol_outer,
        tol_inner,
        verbose,
        log_every,
        decay_start,
        decay_type,
        example_name,
    )  


if __name__ == '__main__':
    main()

