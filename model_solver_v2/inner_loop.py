"""Inner loop, Section 5.1 step 2(b): analytical FONC inversions
(eq. 23-31), the primal/dual primal-dual gradient steps (eq. 32-60), and
the orchestrator tying routing -> freight routing -> inclusive value ->
analytical primal -> gradient ascent -> gradient descent -> convergence
check together for a single outer cycle's inner loop.

Two cross-block timing subtleties, both confirmed directly against the
document's superscripts:

1. Eq. 26/27/32 use a *pre-iteration* ("t") snapshot of xi_ijsb_k /
   q_ijsb / q_ijsb_sk, even though those fields are overwritten to their
   "t+1" value earlier in the same iteration (xi by eq. 28; q_ijsb /
   q_ijsb_sk by the inclusive-value cascade's Step 7). InnerLoopBlock
   snapshots them at the top of each iteration before anything runs.
2. Eq. 32-41 (the primal ascent block) and eq. 42-60 (the dual descent
   block) are each a *single* synchronous vector update (eq. 3-4): every
   equation's RHS reads only pre-this-block ("t") values, never a value
   another equation in the *same* block already advanced to "t+1".
   PrimalGradientBlock / DualGradientBlock therefore read every input
   into a local variable before writing any output back to `state`.
"""
import numpy as np

from .algo_params import AlgoParams
from .convergence import argmax_residual
from .inclusive_value import InclusiveValueBlock
from .params import ModelParams
from .routing import (CommuterRoutingBlock, FreightRoutingBlock,
                       PeripheralRoutingBlock, SecondaryRoutingBlock)
from .state import ModelState

_EPS = 1e-8
# Numerical safeguard for the q^t/q^{t+1} population-share ratios in eq. 26/27:
# these ratios are ~1 near a fixed point, but early iterations' logit shares
# can swing sharply before the cascade stabilizes, occasionally sending a
# ratio's denominator near zero. Clipping bounds the resulting target-price
# spike instead of letting it propagate into an irrecoverable blow-up,
# without affecting the well-behaved (ratio ~ 1) regime the document assumes.
_RATIO_CLIP = 1e3
# rho_j^s (eq. 44, now projected >= 0 -- see DualGradientBlock) enters
# several target-price arguments (eq. 26/27) as an additive component
# alongside other (also nonnegative) terms. A Cobb-Douglas inv_grad
# computes alpha/price, so once a price gets close to 0 the
# functional_forms.py-level 1e-8 floor can still turn it into a ~1e8-scale
# demand spike. Flooring at a small but economically meaningful price here
# (rather than at 1e-8) keeps that inversion well-behaved.
_PRICE_FLOOR = 0.001


class AnalyticalPrimalBlock:
    """Eq. 23-31, executed in the document's literal order so that eq. 28's
    update of xi_ijsb_k is visible to eq. 29-31 but not to eq. 26-27 (which
    use the pre-iteration xi snapshot)."""

    def solve(self, state: ModelState, params: ModelParams, xi_prev, q_ijsb_prev, q_ijsb_sk_prev):
        S, I, Br, A = params.n_sectors, params.n_zones, params.n_b_r, params.n_activities
        net = params.network
        trad = params.is_tradable

        # ---- eq. 23: c_{s'}^{ijsb,t+1} = [grad_c U]^{-1}(p_i^{s',T,t}), s' in S^T ----
        p_T_home = state.p_sT.T[:, trad]  # (I, n_trad) -- p_i^{s',T}
        c_trad_val = params.utility.inv_grad('c', p_T_home)  # (I, n_trad)
        state.c_ijsb_s[..., trad] = c_trad_val[None, :, None, None, :]
        state.c_ijsb_s[..., ~trad] = 0.0

        # ---- eq. 24: d_{s'k}^{ijsb,t+1} = [grad_d u_{s'}]^{-1}(p_k^{s',T,t}), s' in S^T ----
        p_T_k = state.p_sT.T[:, trad]  # (I, n_trad) -- p_k^{s',T}
        d_trad_val = params.activity_utility_T.inv_grad('d', p_T_k)  # (I, n_trad)
        state.d_ijsb_sk[..., trad] = d_trad_val[None, None, None, None, :, :]
        state.d_ijsb_sk[..., ~trad] = 0.0

        # ---- eq. 25: l^{ijsb,t+1} = [grad_l U]^{-1}(r_i^{b,r,t}) ----
        l_val = params.utility.inv_grad('l', state.r_res)  # (I, Br)
        state.l_ijsb[:] = l_val[None, :, None, :]

        # ---- eq. 26: f_h^{ijsb,t+1} ----
        # bracket sum_{n'} q_{in'}^{ijsb,t} reduces to q_ijsb_prev itself for
        # i != j: under all-or-nothing assignment exactly one edge leaves
        # the home node i along this group's shortest path, carrying its
        # full volume; it is 0 when i == j (no trip).
        xi_at_i = _gather_xi_at_i(xi_prev)  # (S,I,I,Br), eq.26's xi_i^{ijsb,t}
        bracket_h = np.where(np.eye(I, dtype=bool)[None, :, :, None], 0.0, q_ijsb_prev)
        q_ratio_h = np.clip(bracket_h / np.maximum(state.q_ijsb, _EPS), 0.0, _RATIO_CLIP)
        target_fh = np.maximum(state.rho_js[:, None, :, None] + xi_at_i * q_ratio_h, _PRICE_FLOOR)
        state.fh_ijsb[:] = params.utility.inv_grad('f_h', target_fh)

        # ---- eq. 27: f_{s'k}^{ijsb,t+1} ----
        # bracket simplifies to q_{s'k}^{ijsb,t} itself: k is purely the
        # destination of the (i,j,s,b,s',k) routing tree, so total inflow
        # into k from this group's own assignment equals its volume and
        # total outflow (within this same routing problem) is zero.
        rho_kb = state.rho_js[:, None, :, None, None, None]
        q_ratio_f = np.clip(q_ijsb_sk_prev / np.maximum(state.q_ijsb_sk, _EPS), 0.0, _RATIO_CLIP)
        xi_term = xi_prev[..., None] * q_ratio_f
        p_N_k = state.p_sN.T  # (I, A)
        target_f = np.zeros((S, I, I, Br, I, A))
        target_f[..., trad] = rho_kb + xi_term[..., trad]
        target_f[..., ~trad] = rho_kb + p_N_k[None, None, None, None, :, ~trad] + xi_term[..., ~trad]
        target_f = np.maximum(target_f, _PRICE_FLOOR)
        f_val = np.zeros((S, I, I, Br, I, A))
        f_val[..., trad] = params.activity_utility_T.inv_grad('f', target_f[..., trad])
        f_val[..., ~trad] = params.activity_utility_N.inv_grad(target_f[..., ~trad])
        state.f_ijsb_sk[:] = f_val

        # ---- eq. 28: xi_k^{ijsb,t+1} = gamma_k^{park,(K)} + rho_j^{s,t} * omega_w * T_k^P(...) ----
        S_k = state.Y_pk.sum(axis=1)  # (I,)
        T_P_val = params.parking_search_time.value(state.yP_k, S_k)  # (I,)
        xi_val = state.gamma_park_k[None, None, :] + params.omega_w * state.rho_js[:, :, None] * T_P_val[None, None, :]
        # xi_val[s,j,k]; broadcast across i,b
        state.xi_ijsb_k[:] = xi_val[:, None, :, None, :]

        # ---- eq. 29: H_j^{s,t+1}, L_j^{sb,t+1} ----
        target_H = state.w_js / np.maximum(state.pi_js, _EPS)
        state.H_js[:] = params.commercial_production.inv_grad('H', target_H, Y=state.Y_js)
        target_L = state.r_nonres[None, :, :] / np.maximum(state.pi_js[:, :, None], _EPS)
        state.L_jsb[:] = params.commercial_production.inv_grad('L', target_L, Y=state.Y_js)

        # ---- eq. 30: Q_j^{ss',t+1}, s' in S^T ----
        target_Q = state.p_sT.T[None, :, trad] / np.maximum(state.pi_js[:, :, None], _EPS)
        Q_val = params.commercial_production.inv_grad('Q', target_Q, Y=state.Y_js)
        state.Q_jss[..., trad] = Q_val
        state.Q_jss[..., ~trad] = 0.0

        # ---- eq. 31: H_j^{p,t+1}, H_{nn'}^{m,t+1}, l_tilde^m_{n,t+1} ----
        target_Hp = params.wage_parking[:, None] / np.maximum(state.pi_jp, _EPS)
        state.H_jp[:] = params.parking_production.inv_grad('H', target_Hp, Y=state.Y_pk)

        target_Hm = state.w_m[:, None] / np.maximum(state.pi_nn_m, _EPS)
        state.H_nn_m[:] = params.transit_vehicle_production.inv_grad(target_Hm, Y=state.V_m_nn)

        r_tilde_nodes = np.zeros(net.n_nodes)
        r_tilde_nodes[:I] = state.r_tilde
        target_l = r_tilde_nodes[None, :] / np.maximum(state.lambda_stat_mn, _EPS)
        state.l_tilde_mn[:] = params.station_capacity.inv_grad(target_l)


def _gather_xi_at_i(xi_arr):
    """xi_arr: HG + (I,) = (S,I,I,Br,I). Returns xi evaluated at k = i
    (the home zone), shape (S,I,I,Br)."""
    S, I, _, Br, _ = xi_arr.shape
    out = np.zeros((S, I, I, Br))
    for i in range(I):
        out[:, i, :, :] = xi_arr[:, i, :, :, i]
    return out


class PrimalGradientBlock:
    """Eq. 32-41: synchronous projected-ascent update of the iterative
    primal vector X."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, xi_prev, t: int):
        S, I = params.n_sectors, params.n_zones
        net = params.network
        a = algo.eta(algo.alpha_x, t)

        h_old = state.h_ijsb.copy()
        l_res_old, l_nonres_old, l_park_old = (state.l_tilde_res.copy(), state.l_tilde_nonres.copy(),
                                                 state.l_tilde_park.copy())
        Y_old = state.Y_js.copy()
        Q_old, Qt_old = state.Q_ij_s.copy(), state.Qtilde_ij_s.copy()
        q0_old, h0_old = state.q0_js.copy(), state.h0_js.copy()
        Qi0_old, Q0i_old = state.Q_i0.copy(), state.Q_0i.copy()
        l_infra_old = state.l_tilde_infra.copy()

        # eq. 32: h^{ijsb,t+1}
        car_term = (xi_prev * state.car_flow_imbalance_ijsb_k).sum(axis=4)
        car_term_per_capita = np.clip(car_term / np.maximum(state.q_ijsb, _EPS), -_RATIO_CLIP, _RATIO_CLIP)
        h_new = np.maximum(0.0, h_old + a * (state.w_js[:, None, :, None] - state.rho_js[:, None, :, None]
                                              - car_term_per_capita))

        # eq. 33: tilde l_i^b, all three families
        l_res_new = np.maximum(0.0, l_res_old + a * (state.r_bar_res - state.r_tilde[:, None]))
        l_nonres_new = np.maximum(0.0, l_nonres_old + a * (state.r_bar_nonres - state.r_tilde[:, None]))
        l_park_new = np.maximum(0.0, l_park_old + a * (state.r_bar_park - state.r_tilde[:, None]))

        # eq. 34: Y_j^{s,t+1}
        price_used = np.where(params.is_tradable[:, None], state.p_sT, state.p_sN)
        Y_new = np.maximum(0.0, Y_old + a * (state.pi_js - price_used))

        # eq. 35-36: Q_{ij}^{s,t+1}, tilde Q_{ij}^{s,t+1}
        nu_at_i = np.zeros((S, I, I))
        nu_at_j = np.zeros((S, I, I))
        for i in range(I):
            i_node = net.zone_to_node(i)
            nu_at_i[:, i, :] = state.nu_s_ij_n[:, i, :, i_node]
        for j in range(I):
            j_node = net.zone_to_node(j)
            nu_at_j[:, :, j] = state.nu_s_ij_n[:, :, j, j_node]
        Q_new = np.maximum(0.0, Q_old + a * (nu_at_i - state.p_sT[:, :, None]))
        Qt_new = np.maximum(0.0, Qt_old + a * (nu_at_j - state.p_sT[:, None, :]))

        # eq. 37: q^{0,js,t+1}
        # routing.py's mu is stored "cost-to-destination" (mu_destination=0,
        # mu_origin=full trip cost -- a single-destination-anchored Dijkstra),
        # whereas eq. 8's FOC inequality (mu_{n'} - mu_n <= cost(n,n')) implies
        # a "cost-from-origin" convention (mu_origin=0, mu_destination=full
        # cost). Since mu_mine(n) + mu_document(n) = full trip cost for any n
        # on the path, every document term (mu_X - mu_Y) becomes (mu_Y - mu_X)
        # when evaluated with the stored (mirrored) values -- i.e. the sign
        # is flipped relative to a literal transcription.
        mu_at_j = np.zeros((S, I))
        mu_at_ng = np.zeros((S, I))
        for j in range(I):
            j_node = net.zone_to_node(j)
            mu_at_j[:, j] = state.mu_0js_n[:, j, j_node]
            mu_at_ng[:, j] = state.mu_0js_n[:, j, net.entry_node]
        q0_new = np.maximum(0.0, q0_old + a * (state.w_js * h0_old - state.lambda0 + (mu_at_j - mu_at_ng)))

        # eq. 38: h^{0,js,t+1} (bracket reduces to gamma_j^{park}*q0_js^t, see module docstring derivation)
        h0_new = np.maximum(0.0, h0_old + a * (state.w_js * q0_old - state.w0 - state.gamma_park_k[None, :] * q0_old))

        # eq. 39-40: Q_{i0}^{s,t+1}, Q_{0i}^{s,t+1}
        Qi0_new = np.maximum(0.0, Qi0_old + a * (state.p_sT - state.theta0 * params.tau_E[:, None]))
        Q0i_new = np.maximum(0.0, Q0i_old + a * (state.theta0 / np.maximum(params.tau_M[:, None], _EPS) - state.p_sT))

        # eq. 41: tilde l_i^{infra,t+1}
        l_infra_new = np.maximum(0.0, l_infra_old + a * (state.r_infra - state.r_tilde))

        state.h_ijsb = h_new
        state.l_tilde_res, state.l_tilde_nonres, state.l_tilde_park = l_res_new, l_nonres_new, l_park_new
        state.Y_js = Y_new
        state.Q_ij_s, state.Qtilde_ij_s = Q_new, Qt_new
        state.q0_js, state.h0_js = q0_new, h0_new
        state.Q_i0, state.Q_0i = Qi0_new, Q0i_new
        state.l_tilde_infra = l_infra_new


class DualGradientBlock:
    """Eq. 42-60: synchronous projected-descent update of the iterative
    dual vector Lambda, including rho_j^s (eq. 44) -- see the projection
    note at that line for why it's projected despite the document's
    literal eq. 44 omitting the \\Proj{...} wrapper other duals carry."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, t: int):
        S, I = params.n_sectors, params.n_zones
        M = params.n_modes_transit
        net = params.network
        a = algo.eta(algo.alpha_lambda, t)

        lambda0_old, w0_old = state.lambda0, state.w0
        rho_old = state.rho_js.copy()
        r_tilde_old = state.r_tilde.copy()
        r_bar_res_old, r_bar_nonres_old, r_bar_park_old = (state.r_bar_res.copy(), state.r_bar_nonres.copy(),
                                                             state.r_bar_park.copy())
        r_res_old, r_nonres_old, r_park_old = state.r_res.copy(), state.r_nonres.copy(), state.r_park.copy()
        r_infra_old = state.r_infra.copy()
        pi_js_old, pi_jp_old, pi_nn_m_old = state.pi_js.copy(), state.pi_jp.copy(), state.pi_nn_m.copy()
        w_js_old, w_m_old = state.w_js.copy(), state.w_m.copy()
        p_sT_old, p_sN_old = state.p_sT.copy(), state.p_sN.copy()
        theta0_old = state.theta0
        lambda_stat_old = state.lambda_stat_mn.copy()

        # eq. 42: lambda^{0,t+1}
        lambda0_new = max(0.0, lambda0_old - a * (params.q_bar_0 - state.q0_js.sum()))

        # eq. 43: w^{0,t+1}
        w0_new = max(0.0, w0_old - a * (params.H_bar - float((state.q0_js * state.h0_js).sum())))

        # eq. 44: rho_j^{s,t+1}. The document's literal eq. 44 omits the
        # \Proj{...} wrapper every other dual carries, but rho_j^s is the
        # multiplier on the household time-budget constraint written as an
        # inequality (time used <= H_bar) -- by standard KKT theory that
        # multiplier must be >= 0 (a negative shadow price on a "<="
        # constraint isn't a valid stationary point), so it is projected
        # here like every other market-clearing dual.
        q_sum_ib = state.q_ijsb.sum(axis=(1, 3))  # sum over i,b -> (S,j)
        qh_sum = (state.q_ijsb * (state.h_ijsb + state.fh_ijsb)).sum(axis=(1, 3))  # (S,j)
        qf_sum = np.einsum('sijbka,sijbka->sj', state.q_ijsb_sk, state.f_ijsb_sk)
        rho_new = np.maximum(0.0, rho_old - a * (q_sum_ib * params.H_bar - qh_sum - qf_sum))

        # eq. 45: tilde r_i^{t+1}
        l_b_sum = state.l_tilde_res.sum(axis=1) + state.l_tilde_nonres.sum(axis=1) + state.l_tilde_park.sum(axis=1)
        l_m_sum = state.l_tilde_mn[:, :I].sum(axis=0)
        r_tilde_new = np.maximum(0.0, r_tilde_old - a * (params.L_bar - l_b_sum - l_m_sum - state.l_tilde_infra))

        # eq. 46: bar r_i^{b,t+1}, all three families
        r_bar_res_new = np.maximum(0.0, r_bar_res_old - a * (state.l_tilde_res - params.alpha_floor_res * state.l_bar_res))
        r_bar_nonres_new = np.maximum(0.0, r_bar_nonres_old - a * (state.l_tilde_nonres - params.alpha_floor_nonres * state.l_bar_nonres))
        r_bar_park_new = np.maximum(0.0, r_bar_park_old - a * (state.l_tilde_park - params.alpha_floor_park * state.l_bar_park))

        # eq. 47: r_i^{b,r,t+1}
        demand_res = np.einsum('sijb,sijb->ib', state.q_ijsb, state.l_ijsb)
        r_res_new = np.maximum(0.0, r_res_old - a * (state.l_bar_res - demand_res))

        # eq. 48: r_j^{b,nr,t+1}
        demand_nonres = state.L_jsb.sum(axis=0)  # (I,Bnr)
        r_nonres_new = np.maximum(0.0, r_nonres_old - a * (state.l_bar_nonres - demand_nonres))

        # eq. 49: r_j^{p,t+1} -- L_j^p := l_bar_park (see params.py docstring); residual is
        # identically zero by this construction, kept for completeness/documentation.
        r_park_new = np.maximum(0.0, r_park_old - a * (state.l_bar_park - state.l_bar_park))

        # eq. 50: r_i^{infra,t+1}
        infra_land = np.zeros(I)
        for e, (u, v) in enumerate(net.edges):
            if u < I:
                infra_land[u] += params.alpha_infra * state.I_nn[e] / 2.0
            if v < I:
                infra_land[v] += params.alpha_infra * state.I_nn[e] / 2.0
        r_infra_new = np.maximum(0.0, r_infra_old - a * (state.l_tilde_infra - infra_land))

        # eq. 51: pi_j^{s,t+1}
        Y_val = params.commercial_production.value(state.H_js, state.L_jsb, state.Q_jss[..., params.is_tradable])
        pi_js_new = np.maximum(0.0, pi_js_old - a * (Y_val - state.Y_js))

        # eq. 52: pi_j^{p,t+1}
        Yp_val = params.parking_production.value(state.H_jp, state.l_bar_park)
        pi_jp_new = np.maximum(0.0, pi_jp_old - a * (Yp_val - state.Y_pk))

        # eq. 53: pi_{nn'}^{m,t+1}
        Ym_val = params.transit_vehicle_production.value(state.H_nn_m)
        pi_nn_m_new = np.maximum(0.0, pi_nn_m_old - a * (Ym_val - state.V_m_nn))

        # eq. 54: w_j^{s,t+1}
        qh_commute = np.einsum('sijb,sijb->sj', state.q_ijsb, state.h_ijsb)
        w_js_new = np.maximum(0.0, w_js_old - a * (qh_commute + state.q0_js * state.h0_js - state.H_js))

        # eq. 55: w_m^{t+1}
        w_m_new = np.maximum(0.0, w_m_old - a * (state.H_nn_m.sum(axis=1) - params.H_allocated_m))

        # eq. 56-57: p_i^{s',T,t+1}
        # Both the at-home (c) and the on-activity (d, evaluated at k = i,
        # the activity taking place at the household's own home zone) terms
        # sum over (s, j, b); see eq. 57's C_i^{total,s'}.
        c_recv = np.einsum('sijb,sijbp->ip', state.q_ijsb, state.c_ijsb_s)
        d_recv = np.zeros((I, S))
        for i in range(I):
            d_recv[i, :] = (state.q_ijsb_sk[:, i, :, :, i, :] * state.d_ijsb_sk[:, i, :, :, i, :]).sum(axis=(0, 1, 2))
        Q_input_recv = state.Q_jss.sum(axis=0)  # sum over producing sector s -> (I, A)
        C_total = c_recv + d_recv + Q_input_recv
        C_total += params.is_gateway[:, None] * state.Q_i0.T

        recv_goods = state.Qtilde_ij_s.sum(axis=1)  # sum over origin -> (S, I) destination-indexed
        p_sT_new = np.maximum(0.0, p_sT_old - a * (
            state.Y_js + recv_goods + params.is_gateway[None, :] * state.Q_0i - C_total.T))

        # eq. 58: p_i^{s',N,t+1}
        f_recv = np.zeros((I, S))
        for i in range(I):
            f_recv[i, :] = (state.q_ijsb_sk[:, i, :, :, i, :] * state.f_ijsb_sk[:, i, :, :, i, :]).sum(axis=(0, 1, 2))
        p_sN_new = np.maximum(0.0, p_sN_old - a * (state.Y_js - f_recv.T))

        # eq. 59: theta_0^{t+1}
        export_term = np.einsum('s,si->', params.tau_E, state.Q_i0 * params.is_gateway[None, :])
        import_term = np.einsum('s,si->', 1.0 / np.maximum(params.tau_M, _EPS), state.Q_0i * params.is_gateway[None, :])
        theta0_new = max(0.0, theta0_old - a * (export_term - import_term))

        # eq. 60: lambda^{stat,t+1}_{m,n}
        F_stat = params.station_capacity.value(state.l_tilde_mn)  # (M, Nn)
        V_out = np.zeros((M, net.n_nodes))
        V_in = np.zeros((M, net.n_nodes))
        for e, (u, v) in enumerate(net.edges):
            V_out[:, u] += state.V_m_nn[:, e]
            V_in[:, v] += state.V_m_nn[:, e]
        lambda_stat_new = np.maximum(0.0, lambda_stat_old - a * (F_stat - V_out - V_in))

        state.lambda0, state.w0 = lambda0_new, w0_new
        state.rho_js = rho_new
        state.r_tilde = r_tilde_new
        state.r_bar_res, state.r_bar_nonres, state.r_bar_park = r_bar_res_new, r_bar_nonres_new, r_bar_park_new
        state.r_res, state.r_nonres, state.r_park = r_res_new, r_nonres_new, r_park_new
        state.r_infra = r_infra_new
        state.pi_js, state.pi_jp, state.pi_nn_m = pi_js_new, pi_jp_new, pi_nn_m_new
        state.w_js, state.w_m = w_js_new, w_m_new
        state.p_sT, state.p_sN = p_sT_new, p_sN_new
        state.theta0 = theta0_new
        state.lambda_stat_mn = lambda_stat_new


class InnerLoopBlock:
    def __init__(self):
        self.commuter = CommuterRoutingBlock()
        self.secondary = SecondaryRoutingBlock()
        self.peripheral = PeripheralRoutingBlock()
        self.freight = FreightRoutingBlock()
        self.cascade = InclusiveValueBlock()
        self.analytical = AnalyticalPrimalBlock()
        self.primal_grad = PrimalGradientBlock()
        self.dual_grad = DualGradientBlock()
        # Cumulative iteration counter shared by *both* alpha_x and
        # alpha_lambda's decay (AlgoParams.eta): persists across every call
        # to solve() over this instance's lifetime. Both step sizes must
        # decay on the *same* basis so their ratio (deliberately set to
        # 10:1, dual slower than primal, to avoid saddle-point oscillation)
        # never drifts -- decaying them on different bases (e.g. one on this
        # cumulative count, the other on the outer cycle K) lets the ratio
        # depend on T_inner*T_msa and can flip which one is actually larger
        # for large enough inner-loop depth. A single shared decay schedule
        # also keeps the *inner* loop's own primal-dual game stable across
        # its many iterations (Robbins-Monro): making alpha_x constant
        # within a cycle (e.g. by decaying it on K alone) was tried and
        # diverged, because it removed exactly this within-cycle damping.
        self.global_t = 0

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams):
        residual_history = []
        for _ in range(algo.T_inner):
            self.global_t += 1
            x_old_norm_inputs = self._snapshot_x(state)

            xi_prev = state.xi_ijsb_k.copy()
            q_ijsb_prev = state.q_ijsb.copy()
            q_ijsb_sk_prev = state.q_ijsb_sk.copy()

            # 2(b)(i): multimodal routing and potentials
            self.commuter.solve(state, params, algo)
            self.secondary.solve(state, params, algo)
            self.peripheral.solve(state, params, algo)

            # 2(b)(ii): freight routing
            self.freight.solve(state, params, algo)

            # 2(b)(iii): inclusive-value cascade + analytical primal forms
            self.cascade.solve(state, params)
            self.analytical.solve(state, params, xi_prev, q_ijsb_prev, q_ijsb_sk_prev)

            # 2(b)(iii)+: gradient ascent-descent step -- both decayed on
            # the same cumulative counter, see __init__'s note.
            self.primal_grad.solve(state, params, algo, xi_prev, self.global_t)
            self.dual_grad.solve(state, params, algo, self.global_t)

            residual = self._residual(state, x_old_norm_inputs)
            residual_history.append(residual)
            if residual['value'] < algo.tol_inner:
                break
        return residual_history

    @staticmethod
    def _snapshot_x(state: ModelState):
        return (state.h_ijsb.copy(), state.Y_js.copy(), state.Q_ij_s.copy(),
                state.q0_js.copy(), state.h0_js.copy())

    @staticmethod
    def _residual(state: ModelState, x_old):
        """Returns {'value','variable','index'} identifying which named
        primal quantity (and which entry within it) is responsible for
        this iteration's max-abs-change -- see convergence.argmax_residual."""
        h_old, Y_old, Q_old, q0_old, h0_old = x_old
        return argmax_residual([
            ('h_ijsb', state.h_ijsb - h_old),
            ('Y_js', state.Y_js - Y_old),
            ('Q_ij_s', state.Q_ij_s - Q_old),
            ('q0_js', state.q0_js - q0_old),
            ('h0_js', state.h0_js - h0_old),
        ])
