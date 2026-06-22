import numpy as np
from .models import ModelParams, ModelState, AlgoParams
from .blocks import TransportBlock, ConvergenceLoggerOuter, ConvergenceLoggerInner


def _mult_step(x, ed, eta, lo=1e-8):
    """
    Multiplicative tâtonnement step (elementwise, works for any array shape).
        x_{t+1} = max(lo, x_t * exp(eta * ed / max(|x_t|, 1)))
    The denominator max(|x|,1) keeps the step from exploding when x is large
    while reducing to the additive update when x is O(1).
    """
    x  = np.asarray(x,  dtype=float)
    ed = np.asarray(ed, dtype=float)
    assert x.shape == ed.shape, f"shape mismatch: x{x.shape} vs ed{ed.shape}"
    exponent = eta * ed / np.maximum(np.abs(x), 1.0)
    exponent = np.clip(exponent, -50.0, 50.0)
    x_new = x * np.exp(exponent)
    x_new = np.nan_to_num(x_new, nan=lo, posinf=1e8, neginf=lo)
    return np.clip(x_new, lo, 1e8)


class ExcessDemand:
    def compute(self, state, params):
        S, I = params.n_sectors, params.n_zones
        net  = params.network

        # ---- Goods: ED^s_i = sum_{s',j} c^{s,s'}_{ij} q^{s'}_{ij}
        #                    + sum_{s'} Q^{s,s'}_i
        #                    + sum_{j != i} Q^s_{ij}
        #                    - Y^s_i
        #                    - sum_{j != i} Q_tilde^s_{ji}
        # state.c shape: (good=S, sector=S, I, J) with c[k, s, i, j] = c^{s,k}_{ij}
        # consumption of good k at zone i: sum over s and j of c[k,s,i,j] * q_pop[s,i,j]
        cons = np.einsum('ksij,sij->ki', state.c, state.q_pop)         # (S goods, I)
        # intersectoral input of good k used at zone j by all sectors s: sum_s Q_inter[s, k, j]
        # Q_inter[s, k, j]: good k consumed by sector s in zone j -> sum over s gives demand of k at j
        inter = state.Q_inter.sum(axis=0)                              # (S goods, J)

        # outgoing shipments of good s from i: sum_{j != i} Q_ship[s,i,j]
        ship_out = state.Q_ship.sum(axis=2) - np.diagonal(state.Q_ship, axis1=1, axis2=2)
        # incoming received at i: sum_{j != i} Q_tilde[s, j, i]
        recv_in  = state.Q_tilde.sum(axis=1) - np.diagonal(state.Q_tilde, axis1=1, axis2=2)

        self.ED_goods = cons + inter + ship_out - state.Y - recv_in     # (S, I)

        # ---- Time: ED_sj = sum_{i} [ q^s_{ij} h^s_{ij} + d_w sum_{nn'} q^s_{ij,nn'} T_{nn'} ] - q^s_ij * Hbar
        T_edge = params.t_link(state.Q_hat, state.I_infra)  # (n_edges,)
        # h^s_{ij} satisfies H^s_j = sum_i q^s_{ij} h^s_{ij}.
        # Approximate at aggregate level: total commuting hours per sector = H_prod[s,:].sum() (labor delivered).
        H_sj = state.H_prod
        commute_time_sj  = params.d_w * (state.q_hat_sj * T_edge[None, None, :]).sum(axis=2) # (S, I)
        q_sj = state.q_pop.sum(axis=1)
        self.ED_time = (H_sj + commute_time_sj - q_sj * params.H_bar) / (params.H_bar * (params.q_bar/params.n_sectors*params.n_zones))  # (S, I)

        # ---- Land: ED_i = sum_s L^s_i + sum_{s,j} q^s_{ij} l^s_{ij} - Lbar_i
        prod_land = state.L_prod.sum(axis=0)                            # (I,)
        res_land  = (state.q_pop * state.l_res).sum(axis=(0, 2))        # sum over s and j -> (I,)
        self.ED_land = (prod_land + res_land - params.L_bar)/params.L_bar  # (I,)
        

        # ---- Infrastructure budget
        self.ED_infra = float((net.kappa * state.I_infra).sum() - params.K) # scalar, normalized by K

class StoppingCriterion:
    def transport(self, state: ModelState, gamma_old, params: ModelParams) -> dict:

        eps_Qhat = 1E-8
        eps_gamma = 1E-8

        return {
            "link_flow" : eps_Qhat,
            "gamma_fp"  : eps_gamma,
        }

    def allocation(self, state, params):

        # ---- Goods: ED^s_i = sum_{s',j} c^{s,s'}_{ij} q^{s'}_{ij}
        #                    + sum_{s'} Q^{s,s'}_i
        #                    + sum_{j != i} Q^s_{ij}
        #                    - Y^s_i
        #                    - sum_{j != i} Q_tilde^s_{ji}
        # state.c shape: (good=S, sector=S, I, J) with c[k, s, i, j] = c^{s,k}_{ij}
        # consumption of good k at zone i: sum over s and j of c[k,s,i,j] * q_pop[s,i,j]
        cons = np.einsum('ksij,sij->ki', state.c, state.q_pop)         # (S goods, I)
        # intersectoral input of good k used at zone j by all sectors s: sum_s Q_inter[s, k, j]
        # Q_inter[s, k, j]: good k consumed by sector s in zone j -> sum over s gives demand of k at j
        inter = state.Q_inter.sum(axis=0)                              # (S goods, J)

        # outgoing shipments of good s from i: sum_{j != i} Q_ship[s,i,j]
        ship_out = state.Q_ship.sum(axis=2) - np.diagonal(state.Q_ship, axis1=1, axis2=2)
        # incoming received at i: sum_{j != i} Q_tilde[s, j, i]
        recv_in  = state.Q_tilde.sum(axis=1) - np.diagonal(state.Q_tilde, axis1=1, axis2=2)

        eps_goods =np.max(np.array(cons + inter + ship_out - state.Y - recv_in))/np.mean(state.Y)
        # 2. TIME CONSTRAINT -> drives w update
        # ED[time](s) = H_prod.sum(i) + total_commute - q_bar_s * H_bar
        # q_hat_sj shape (S, I, n_edges), T_edge computed from current state
        T_edge = params.t_link(state.Q_hat, state.I_infra)         # (n_edges,)
        total_commute = params.d_w * np.sum(
            state.q_hat_sj * T_edge[np.newaxis, np.newaxis, :],
            axis=(1, 2)
        )                                                           # (S,): sum over i and edges
        ED_time  = state.H_prod.sum(axis=1) + total_commute - state.q_bar_s * params.H_bar  # (S,)
        eps_time = np.max(np.abs(ED_time)) / (params.q_bar * params.H_bar)

        # 3. LAND MARKET -> drives r update
        # ED[land] = L_prod + residential_land - L_bar
        res_land  = np.sum(state.q_pop * state.l_res, axis=(0, 2))  # (I,): sum over s,j
        prod_land = np.sum(state.L_prod, axis=0)                    # (J,)
        ED_land   = prod_land + res_land - params.L_bar              # (J,)
        eps_land  = np.max(np.abs(ED_land) / params.L_bar)
    
        return {
        "goods" : eps_goods,
        "time"  : eps_time,
        "land"  : eps_land}
    
    def infrastructure(self, state, params):
    # 4. INFRASTRUCTURE BUDGET -> drives beta update
        # Two regimes depending on whether beta is effectively zero
        beta_tol = 1e-8
        g = state.gamma_I - state.beta * params.network.kappa   # shape: (n_edges,)

        tol = 1e-6

        eps_infra = g

        return {"infra": eps_infra}
    
    def global_convergence(self, state, params, hyperparams):
        alloc = self.allocation(state, params)
        conv_allocation = max(alloc.values()) < hyperparams.tol_outer
        transp = self.transport(state, state.gamma, params)
        conv_transport = max(transp.values()) < hyperparams.tol_inner
        infra = self.infrastructure(state, params)

        conv_infra = False
        

        fully_converged = conv_allocation and conv_infra and conv_transport

        

        return {
            'allocation' : alloc, 
            'transport': transp,
            'infra': infra,
        }


class PriceUpdater:
    def __init__(self, excess_demand_block: ExcessDemand):
        self.ed = excess_demand_block

    def update_prices(self, state: ModelState, params: ModelParams, hyperparams: AlgoParams, t:int = 0):
        # vectors / matrices: all handled elementwise
        state.p = _mult_step(state.p, self.ed.ED_goods, hyperparams.eta(hyperparams.eta_p, t))   # (S, I)
        state.w = _mult_step(state.w, self.ed.ED_time,  hyperparams.eta(hyperparams.eta_w, t))   # (S,)
        state.r = _mult_step(state.r, self.ed.ED_land,  hyperparams.eta(hyperparams.eta_r, t))   # (I,)

    def update_beta(self, state: ModelState, params: ModelParams, hyperparams:AlgoParams, t:int =0):
        # beta is a scalar — project on R_+
        exponent = hyperparams.eta(hyperparams.eta_beta, t) * self.ed.ED_infra / np.maximum(state.beta, 1.0)
        beta_new = state.beta * np.exp(exponent)
        state.beta = beta_new


class TransportSolver:
    def __init__(self, transport_block: TransportBlock):
        self.transport_block = transport_block
        self.logger = ConvergenceLoggerInner()
        self.stopping_criterion = StoppingCriterion()

    def solve(self, state, params, hyperparams, t_outer):
        for t in range(hyperparams.T_inner):
            global_t = t_outer * hyperparams.T_inner + t
            Q_hat_old = state.Q_hat.copy()
            gamma_old = state.gamma.copy()
            self.transport_block.solve(state, params, hyperparams, global_t)
            dQ = np.linalg.norm(state.Q_hat - Q_hat_old)
            dg = np.linalg.norm(state.gamma - gamma_old)/np.mean(state.gamma)
            self.logger.log(dQ, dg)
            eps = self.stopping_criterion.transport(state, gamma_old, params)
            if max(dg, dQ) < hyperparams.tol_inner:
                break
            if dQ > 10**4 or dg > 10**4:  # sanity check for divergence
                print(state.gamma)
                raise ValueError(f"Inner loop diverging at outer iter {t_outer}, inner iter {t}: dQ={dQ:.4e}, dg={dg:.4e}")

class AllocationSolver:
    def __init__(self, household_block, production_block, transport_block, excess_demand_block, price_updater):
        self.household_block = household_block
        self.production_block = production_block
        self.transport_block = transport_block
        self.excess_demand_block = excess_demand_block
        self.price_updater = price_updater
        self.transport_solver = TransportSolver(transport_block)
        self.logger_alloc = ConvergenceLoggerOuter()
        self.stopping_criterion = StoppingCriterion()
    
    def solve(self, initial_state, params, hyperparams):
        state = initial_state
        for t in range(hyperparams.T_allocation):
            p_old, w_old, r_old = (
                state.p.copy(), state.w.copy(), state.r.copy()
            )

            self.household_block.solve(state, params)
            self.production_block.solve(state, params)
            self.transport_solver.solve(state, params, hyperparams,t)

            self.excess_demand_block.compute(state, params)
            self.price_updater.update_prices(state, params, hyperparams, t)

            ed_norm = float(np.linalg.norm(self.excess_demand_block.ED_goods)
                          + np.linalg.norm(self.excess_demand_block.ED_time)
                          + np.linalg.norm(self.excess_demand_block.ED_land))
            dprice = float(np.linalg.norm(state.p - p_old)
                         + np.linalg.norm(state.w - w_old)
                         + np.linalg.norm(state.r - r_old))
            
            #if hyperparams.verbose and t % hyperparams.log_every == 0:
                #print(f"  [alloc {t:4d}] |ED|={ed_norm:.4e}  |Δprice|={dprice:.4e}")

            prices= [state.p, state.w, state.r]
            production= state.Y
            aggregate_flows = state.Q_hat
            global_conv = self.stopping_criterion.global_convergence(state, params, hyperparams)
            self.logger_alloc.log(ed_norm, dprice, prices, production, aggregate_flows, global_conv)
            eps = self.stopping_criterion.allocation(state, params)
            #if max(eps.values()) < hyperparams.tol_outer:
            #    break
        return state



class UrbanModelSolver:
    def __init__(self, household_block, production_block, transport_block,
                 infrastructure_block, excess_demand_block, price_updater):
        self.household_block      = household_block
        self.production_block     = production_block
        self.transport_block      = transport_block
        self.infrastructure_block = infrastructure_block
        self.excess_demand_block  = excess_demand_block
        self.price_updater        = price_updater
        self.allocation_loop_solver = AllocationSolver(household_block, production_block, transport_block, excess_demand_block, price_updater)
        self.logger               = ConvergenceLoggerOuter()
        self.stopping_criterion = StoppingCriterion()

    def solve(self, initial_state, params, hyperparams):
        state = initial_state
        for t in range(hyperparams.T_outer):
            p_old, w_old, r_old, beta_old = (
                state.p.copy(), state.w.copy(), state.r.copy(), state.beta
            )

            self.allocation_loop_solver.solve(state, params, hyperparams)
            self.infrastructure_block.update(state, params, hyperparams, t)
            
            self.excess_demand_block.compute(state, params)
            self.price_updater.update_beta(state, params, hyperparams, t)
            global_conv = self.stopping_criterion.global_convergence(state, params, hyperparams)

            residuals = float(np.linalg.norm(self.excess_demand_block.ED_goods)
                          + np.linalg.norm(self.excess_demand_block.ED_time)
                          + np.linalg.norm(self.excess_demand_block.ED_land)
                          + np.linalg.norm(np.array(list(global_conv['infra'].values())))
            )

            dprice = float(np.linalg.norm(state.p - p_old)
                         + np.linalg.norm(state.w - w_old)
                         + np.linalg.norm(state.r - r_old)
                         + abs(state.beta - beta_old))
            prices= [state.p, state.w, state.r]
            production= state.Y
            aggregate_flows = state.Q_hat
            global_conv = self.stopping_criterion.global_convergence(state, params, hyperparams)
            self.logger.log(residuals, dprice, prices, production, aggregate_flows, global_conv)
            
            if hyperparams.verbose and t % hyperparams.log_every == 0:
                print(f"[outer {t:4d}] |ED|={residuals:.4e}  |Δprice|={dprice:.4e}")
            
            
            
        print(global_conv)
        return state, global_conv

