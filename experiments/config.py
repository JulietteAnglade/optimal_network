import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, replace
from model_solver import *

@dataclass
class ExperimentConfig:
    name: str

    # model parameters
    n_locations: int
    n_modes: int
    types_of_modes: list
    distance_matrix: any
    current_infra_level: any
    capacity: any
    constraints: any

    n_sectors: int
    production_function: any
    utility_function: any

    L_bar: float
    H_bar: float
    q_bar: float
    K: float

    congestion_a: float   
    congestion_b: float    

    tau_link: TauLink
    t_link: TimeLink
    sector_dependance_tau: any
    d_w: float
    phi: float
    sigma: float
    sigma_tilde: float

    # solver hyperparameters
    T_outer: int
    eta_p: float
    eta_w: float
    eta_r: float
    T_allocation: int
    eta_beta: float
    eta_I: float
    eta_H: float
    T_inner: int
    alpha_Q: float
    alpha_Qhat: float
    eta_gamma: float
    tol_outer: float
    tol_inner: float
    verbose: bool
    log_every: int
    decay_start: int
    decay_type: str

    