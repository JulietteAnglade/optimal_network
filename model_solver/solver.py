import numpy as np
from .models import ModelParams, ModelState, AlgoParams
from .blocks import OuterLoopBlock


def _mult_step(x, ed, eta, lo=1e-8):
    x = np.asarray(x, dtype=float)
    ed = np.asarray(ed, dtype=float)
    assert x.shape == ed.shape, f"shape mismatch: x{x.shape} vs ed{ed.shape}"
    exponent = eta * ed / np.maximum(np.abs(x), 1.0)
    exponent = np.clip(exponent, -50.0, 50.0)
    x_new = x * np.exp(exponent)
    x_new = np.nan_to_num(x_new, nan=lo, posinf=1e8, neginf=lo)
    return np.clip(x_new, lo, 1e8)


class ExcessDemand:
    def compute(self, state, params):
        cons = np.einsum('ksij,sij->ki', state.c, state.q_pop)
        inter = state.Q_inter.sum(axis=0)
        ship_out = state.Q_ship.sum(axis=2) - np.diagonal(state.Q_ship, axis1=1, axis2=2)
        recv_in = state.Q_tilde.sum(axis=1) - np.diagonal(state.Q_tilde, axis1=1, axis2=2)
        self.ED_goods = cons + inter + ship_out - state.Y - recv_in
        T_edge = params.t_link(state.Q_hat, state.I_infra)
        commute_time = params.d_w * (state.q_hat_sj * T_edge[None, None, :]).sum(axis=2)
        self.ED_time = state.H_prod + commute_time - state.q_pop.sum(axis=1) * params.H_bar
        prod_land = state.L_prod.sum(axis=0)
        res_land = (state.q_pop * state.l_res).sum(axis=(0, 2))
        self.ED_land = prod_land + res_land - params.L_bar
        self.ED_infra = float(np.dot(params.network.kappa, state.I_infra) - params.K)


class PriceUpdater:
    def __init__(self, excess_demand_block: ExcessDemand):
        self.ed = excess_demand_block

    def update_prices(self, state: ModelState, params: ModelParams, hyperparams: AlgoParams, t: int = 0):
        state.p = _mult_step(state.p, self.ed.ED_goods, hyperparams.eta(hyperparams.eta_p, t))
        state.w = _mult_step(state.w, self.ed.ED_time, hyperparams.eta(hyperparams.eta_w, t))
        state.r = _mult_step(state.r, self.ed.ED_land, hyperparams.eta(hyperparams.eta_r, t))

    def update_beta(self, state: ModelState, params: ModelParams, hyperparams: AlgoParams, t: int = 0):
        exponent = hyperparams.eta(hyperparams.eta_beta, t) * self.ed.ED_infra / np.maximum(state.beta, 1.0)
        beta_new = state.beta * np.exp(exponent)
        state.beta = np.maximum(beta_new, 0.0)


class UrbanModelSolver:
    def __init__(self, household_block=None, production_block=None, transport_block=None,
                 infrastructure_block=None, excess_demand_block=None, price_updater=None):
        self.household_block = household_block
        self.production_block = production_block
        self.transport_block = transport_block
        self.infrastructure_block = infrastructure_block
        self.excess_demand_block = excess_demand_block
        self.price_updater = price_updater
        self.outer_block = OuterLoopBlock()
        self.logger = self.outer_block.logger

    def solve(self, initial_state, params, hyperparams):
        state = initial_state
        state = self.outer_block.solve(state, params, hyperparams)
        return state
