import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run import run
from model_solver import *
from config import ExperimentConfig
from dataclasses import replace
from itertools import product


def solve_for_experiment(config: ExperimentConfig):
    
    params = None

    
    
    final_state = run(
        n_locations=config.n_locations,
        n_modes=config.n_modes,
        types_of_modes=config.types_of_modes,
        distance_matrix=config.distance_matrix,
        current_infra_level=config.current_infra_level,
        capacity=config.capacity,
        constraints=config.constraints,
        n_sectors=config.n_sectors,
        production_function=config.production_function,
        utility_function=config.utility_function,
        L_bar=config.L_bar,
        H_bar=config.H_bar,
        q_bar=config.q_bar,
        K=config.K,
        sector_dependance_tau=config.sector_dependance_tau,
        tau_link=config.tau_link,
        t_link=config.t_link,
        d_w=config.d_w,
        phi=config.phi,
        sigma=config.sigma,
        sigma_tilde=config.sigma_tilde,
        T_outer=config.T_outer,
        eta_p=config.eta_p,
        eta_w=config.eta_w,
        eta_r=config.eta_r,
        T_allocation=config.T_allocation,
        eta_beta=config.eta_beta,
        eta_I=config.eta_I,
        eta_H=config.eta_H,
        T_inner=config.T_inner,
        alpha_Q=config.alpha_Q,
        alpha_Qhat=config.alpha_Qhat,
        eta_gamma=config.eta_gamma,
        tol_outer=config.tol_outer,
        tol_inner=config.tol_inner,
        verbose=config.verbose,
        log_every=config.log_every,
        decay_start=config.decay_start,
        decay_type=config.decay_type,
        example_name=config.name,
        congestion_a=config.congestion_a,
        congestion_b=config.congestion_b
    )

    return final_state, params



class ExperimentRunner:

    def __init__(self, base_config):
        self.base = base_config

    def run_grid(self, param_grid):

        results = []

        keys = param_grid.keys()

        for values in product(*param_grid.values()):

            updates = dict(zip(keys, values))

            config = replace(
                self.base,
                **updates,
                name="_".join(
                    f"{k}={v}"
                    for k,v in updates.items()
                )
            )

            print(f"Running {config.name}")

            state, params = solve_for_experiment(config)

            results.append({
                "config": config,
                "state": state,
                "params": params
            })

        return results

class ExperimentVisualizer:

    @staticmethod
    def compare_experiments(results):

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12,10)
        )

        for ax, r in zip(
            axes.flatten(),
            results
        ):

            population = r["state"].L.reshape(5,5)

            im=ax.imshow(population)

            ax.set_title(
                r["config"].name
            )

            plt.colorbar(im,ax=ax)

        plt.tight_layout()

import pickle
from pathlib import Path

class ResultsManager:

    @staticmethod
    def save(results, folder):

        Path(folder).mkdir(exist_ok=True)

        with open(
            f"{folder}/results.pkl",
            "wb"
        ) as f:

            pickle.dump(results,f)

    @staticmethod
    def load(path):

        with open(path,"rb") as f:
            return pickle.load(f)