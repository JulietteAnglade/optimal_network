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
    n_locations = 2
    n_modes = 2

    types_of_modes = [
        ModeOfTransport(name='Car', speed=50.0, cost_per_km=1.2),
        ModeOfTransport(name='Train', speed=90.0, cost_per_km=0.45),
    ]

    distance_matrix = np.array([
        [0.0, 25.0],
        [25.0, 0.0],
    ])
    current_infra_level = np.array([
        [[2.0, 1.0], [1.0, 2.0]],
        [[3.0, 2.5], [2.5, 3.0]],
    ])
    capacity = np.array([
        [[12.0, 6.0], [6.0, 12.0]],
        [[30.0, 20.0], [20.0, 30.0]],
    ])
    constraints = np.array([
        [1, 1],  # goods_allowed, commuters_allowed for car
        [1, 1],  # commuters_allowed for train
    ], dtype=bool)

    # ----------------------- 2. model parameters -----------------------
    n_sectors = 2

    production_function = CobbDouglasProduction(
        A=np.array([[1.35, 0.95], [0.90, 1.30]]),
        aH=np.array([0.45, 0.44]),
        aL=np.array([0.16, 0.16]),
        aQ=np.array([[0.0, 0.24], [0.24, 0.0]]),
    )

    utility_function = CobbDouglasUtility(
        alpha_c=np.array([[0.55, 0.25], [0.25, 0.55]]),
        alpha_l=np.array([0.15, 0.15]),
        alpha_f=np.array([0.1, 0.1]),
    )

    L_bar = np.array([3.0, 4.0])
    H_bar = 1.0
    q_bar = 20.0
    K = 300

    sector_dependance_tau = np.array([0.015, 0.035])
    tau_link = IcebergTau
    t_link = BPRTime

    d_w = 1.0
    phi = np.array([0.8, 1.4])
    sigma = np.array([0.45, 0.45])
    sigma_tilde = 0.60

    # ----------------- 3. algo hyperparameters ------------------------
    T_outer = 150000
    eta_p = 0.01
    eta_w = 0.01
    eta_r = 0.01
    T_allocation = 1
    eta_beta = 0.01
    eta_I = 0.025
    eta_H = 0.01
    T_inner = 1000
    alpha_Q = 0.01
    alpha_Qhat = 0.01
    eta_gamma = 0.3
    tol_outer = 5e-3
    tol_inner = 1e-4
    verbose = True
    log_every = 50
    decay_start = 0
    decay_type = None

    example_name = 'test_exampleBIS'
    
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



    # --------------------- 1. real world network ---------------------
    n_locations = 2
    n_modes = 2

    types_of_modes = [
        ModeOfTransport(name='Car', speed=50.0, cost_per_km=1.2),
        ModeOfTransport(name='Train', speed=90.0, cost_per_km=0.45),
    ]

    distance_matrix = np.array([
        [0.0, 25.0],
        [25.0, 0.0],
    ])
    current_infra_level = np.array([
        [[2.0, 1.0], [1.0, 2.0]],
        [[3.0, 2.5], [2.5, 3.0]],
    ])
    capacity = np.array([
        [[12.0, 6.0], [6.0, 12.0]],
        [[30.0, 20.0], [20.0, 30.0]],
    ])
    constraints = np.array([
        [1, 1],  # goods_allowed, commuters_allowed for car
        [0, 1],  # commuters_allowed for train
    ], dtype=bool)

    # ----------------------- 2. model parameters -----------------------
    n_sectors = 2

    production_function = CobbDouglasProduction(
        A=np.array([[1.35, 0.95], [0.90, 1.30]]),
        aH=np.array([0.45, 0.44]),
        aL=np.array([0.16, 0.16]),
        aQ=np.array([[0.0, 0.24], [0.24, 0.0]]),
    )

    utility_function = CobbDouglasUtility(
        alpha_c=np.array([[0.55, 0.25], [0.25, 0.55]]),
        alpha_l=np.array([0.15, 0.15]),
        alpha_f=np.array([0.1, 0.1]),
    )

    L_bar = np.array([3.0, 4.0])
    H_bar = 1.0
    q_bar = 20.0
    K = 300

    sector_dependance_tau = np.array([0.015, 0.035])
    tau_link = IcebergTau
    t_link = BPRTime

    d_w = 1.0
    phi = np.array([0.8, 1.4])
    sigma = np.array([0.45, 0.45])
    sigma_tilde = 0.60

    # ----------------- 3. algo hyperparameters ------------------------
    T_outer = 150000
    eta_p = 0.01
    eta_w = 0.01
    eta_r = 0.01
    T_allocation = 1
    eta_beta = 0.01
    eta_I = 0.025
    eta_H = 0.01
    T_inner = 1000
    alpha_Q = 0.01
    alpha_Qhat = 0.01
    eta_gamma = 0.3
    tol_outer = 5e-3
    tol_inner = 1e-4
    verbose = True
    log_every = 50
    decay_start = 0
    decay_type = None

    example_name = 'test_exampleBIS1'
    
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



    n_locations = 2
    n_modes = 2

    types_of_modes = [
        ModeOfTransport(name='Car', speed=50.0, cost_per_km=1.2),
        ModeOfTransport(name='Train', speed=90.0, cost_per_km=0.45),
    ]

    distance_matrix = np.array([
        [0.0, 25.0],
        [25.0, 0.0],
    ])
    current_infra_level = np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ])
    capacity = np.array([
        [[12.0, 20.0], [20.0, 12.0]],
        [[30.0, 20.0], [20.0, 30.0]],
    ])
    constraints = np.array([
        [1, 1],  # goods_allowed, commuters_allowed for car
        [1, 1],  # commuters_allowed for train
    ], dtype=bool)

    # ----------------------- 2. model parameters -----------------------
    n_sectors = 2

    production_function = CobbDouglasProduction(
        A=np.array([[1.35, 0.95], [0.90, 1.30]]),
        aH=np.array([0.45, 0.44]),
        aL=np.array([0.16, 0.16]),
        aQ=np.array([[0.0, 0.24], [0.24, 0.0]]),
    )

    utility_function = CobbDouglasUtility(
        alpha_c=np.array([[0.55, 0.25], [0.25, 0.55]]),
        alpha_l=np.array([0.15, 0.15]),
        alpha_f=np.array([0.1, 0.1]),
    )

    L_bar = np.array([3.0, 4.0])
    H_bar = 1.0
    q_bar = 20.0
    K = 300

    sector_dependance_tau = np.array([0.015, 0.035])
    tau_link = IcebergTau
    t_link = BPRTime

    d_w = 1.0
    phi = np.array([0.8, 1.4])
    sigma = np.array([0.45, 0.45])
    sigma_tilde = 0.60

    # ----------------- 3. algo hyperparameters ------------------------
    T_outer = 150000
    eta_p = 0.01
    eta_w = 0.01
    eta_r = 0.01
    T_allocation = 1
    eta_beta = 0.01
    eta_I = 0.025
    eta_H = 0.01
    T_inner = 1000
    alpha_Q = 0.01
    alpha_Qhat = 0.01
    eta_gamma = 0.3
    tol_outer = 5e-3
    tol_inner = 1e-4
    verbose = True
    log_every = 50
    decay_start = 0
    decay_type = None

    example_name = 'test_exampleBIS2'
    
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


    n_locations = 2
    n_modes = 2

    types_of_modes = [
        ModeOfTransport(name='Car', speed=50.0, cost_per_km=0.45),
        ModeOfTransport(name='Train', speed=100.0, cost_per_km=1.2),
    ]

    distance_matrix = np.array([
        [0.0, 25.0],
        [25.0, 0.0],
    ])
    current_infra_level = np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ])
    capacity = np.array([
        [[12.0, 20.0], [20.0, 12.0]],
        [[30.0, 20.0], [20.0, 30.0]],
    ])
    constraints = np.array([
        [1, 1],  # goods_allowed, commuters_allowed for car
        [1, 1],  # commuters_allowed for train
    ], dtype=bool)

    # ----------------------- 2. model parameters -----------------------
    n_sectors = 2

    production_function = CobbDouglasProduction(
        A=np.array([[1.35, 0.95], [0.90, 1.30]]),
        aH=np.array([0.45, 0.44]),
        aL=np.array([0.16, 0.16]),
        aQ=np.array([[0.0, 0.24], [0.24, 0.0]]),
    )

    utility_function = CobbDouglasUtility(
        alpha_c=np.array([[0.55, 0.25], [0.25, 0.55]]),
        alpha_l=np.array([0.15, 0.15]),
        alpha_f=np.array([0.1, 0.1]),
    )

    L_bar = np.array([3.0, 4.0])
    H_bar = 1.0
    q_bar = 20.0
    K = 300

    sector_dependance_tau = np.array([0.015, 0.035])
    tau_link = IcebergTau
    t_link = BPRTime

    d_w = 1.0
    phi = np.array([0.8, 1.4])
    sigma = np.array([0.45, 0.45])
    sigma_tilde = 0.60

    # ----------------- 3. algo hyperparameters ------------------------
    T_outer = 150000
    eta_p = 0.01
    eta_w = 0.01
    eta_r = 0.01
    T_allocation = 1
    eta_beta = 0.01
    eta_I = 0.025
    eta_H = 0.01
    T_inner = 1000
    alpha_Q = 0.01
    alpha_Qhat = 0.01
    eta_gamma = 0.3
    tol_outer = 5e-3
    tol_inner = 1e-4
    verbose = True
    log_every = 50
    decay_start = 0
    decay_type = None

    example_name = 'test_exampleBIS3'
    print(example_name)
    
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

