import numpy as np
import heapq
from .models import ModelParams, ModelState, AlgoParams
from scipy.special import logsumexp


class ConvergenceLoggerOuter:
    def __init__(self):
        self.history = {
            'residuals': [],
            'price_changes': [],
            'prices': [],
            'production': [],
            'aggregate_flows': [],
            'global_conv': []
        }

    def log(self, residuals, price_changes, prices, production, aggregate_flows, global_conv):
        self.history['residuals'].append(residuals)
        self.history['price_changes'].append(price_changes)
        self.history['prices'].append([
            np.array(prices[0], copy=True),
            np.array(prices[1], copy=True),
            np.array(prices[2], copy=True),
        ])
        self.history['production'].append(np.array(production, copy=True))
        self.history['aggregate_flows'].append(np.array(aggregate_flows, copy=True))
        self.history['global_conv'].append(global_conv)


class ConvergenceLoggerInner:
    def __init__(self):
        self.history = {
            'flow_changes': [],
            'congestion_changes': []
        }

    def log(self, flow_changes, congestion_changes):
        self.history['flow_changes'].append(flow_changes)
        self.history['congestion_changes'].append(congestion_changes)


def _project_nonnegative(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def _project_bounds(x, lower, upper):
    x = np.asarray(x, dtype=float)
    return np.minimum(np.maximum(x, lower), upper)


def _ensure_extended_state(state, params):
    if hasattr(state, 'initialize_extended'):
        state.initialize_extended(params)


def _fast_dijkstra(n_nodes, src_nodes, weights, indptr, indices, edge_ids):
    dist = np.full(n_nodes, np.inf)
    pred_node = np.full(n_nodes, -1, dtype=np.int32)
    pred_edge = np.full(n_nodes, -1, dtype=np.int32)
    pq = []
    for u in src_nodes:
        dist[u] = 0.0
        heapq.heappush(pq, (0.0, int(u)))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for i in range(indptr[u], indptr[u + 1]):
            v = indices[i]
            e = edge_ids[i]
            nd = d + weights[e]
            if nd < dist[v]:
                dist[v] = nd
                pred_node[v] = u
                pred_edge[v] = e
                heapq.heappush(pq, (nd, int(v)))
    return dist, pred_node, pred_edge


def _fast_accumulate(S, I, n_nodes, n_edges, q_pop, best_dest, pred_node, pred_edge, zone_matrix):
    new_q = np.zeros((S, I, n_edges))
    for s in range(S):
        for i in range(I):
            src_nodes = zone_matrix[i]
            is_src = np.zeros(n_nodes, dtype=bool)
            for src in src_nodes:
                if src >= 0:
                    is_src[src] = True
            for j in range(I):
                if i == j:
                    continue
                vol = q_pop[s, i, j]
                if vol <= 0:
                    continue
                cur = best_dest[s, j, i]
                if cur < 0:
                    continue
                while cur >= 0 and not is_src[cur]:
                    e = pred_edge[s, j, i, cur]
                    if e < 0:
                        break
                    new_q[s, j, e] += vol
                    cur = pred_node[s, j, i, cur]
    return new_q


def _fast_bellman_ford(n_nodes, dest_nodes, p_sj, tau_s, gamma_phi, u_arr, v_arr, e_arr):
    v = np.full(n_nodes, -np.inf)
    nxt_node = np.full(n_nodes, -1, dtype=np.int32)
    nxt_edge = np.full(n_nodes, -1, dtype=np.int32)
    for d in dest_nodes:
        if d >= 0:
            v[d] = p_sj
    for _ in range(n_nodes - 1):
        updated = False
        for idx in range(len(e_arr)):
            pred = u_arr[idx]
            node = v_arr[idx]
            edge = e_arr[idx]
            if not np.isfinite(v[node]):
                continue
            cand = tau_s[edge] * v[node] - gamma_phi[edge]
            if cand > v[pred]:
                v[pred] = cand
                nxt_node[pred] = node
                nxt_edge[pred] = edge
                updated = True
        if not updated:
            break
    return v, nxt_node, nxt_edge


def _fast_compute_surplus(S, I, n_nodes, zone_matrix, v, nxt_node, nxt_edge, tau, p):
    nu = np.full((S, I, I), -np.inf)
    tau_eff = np.zeros((S, I, I))
    active = np.zeros((S, I, I), dtype=bool)
    for s in range(S):
        for j in range(I):
            dst_nodes = zone_matrix[j]
            is_dst = np.zeros(n_nodes, dtype=bool)
            for d in dst_nodes:
                if d >= 0:
                    is_dst[d] = True
            for i in range(I):
                if i == j:
                    continue
                best_val = -np.inf
                best_src = -1
                for src in zone_matrix[i]:
                    if src >= 0 and v[s, j, src] > best_val:
                        best_val = v[s, j, src]
                        best_src = src
                if best_src == -1:
                    continue
                nu[s, i, j] = best_val
                active[s, i, j] = best_val >= p[s, i]
                if nxt_edge[s, j, best_src] < 0:
                    continue
                t_eff = 1.0
                cur = best_src
                steps = 0
                while cur >= 0 and not is_dst[cur] and steps < n_nodes:
                    e = nxt_edge[s, j, cur]
                    if e < 0:
                        break
                    t_eff *= tau[s, e]
                    cur = nxt_node[s, j, cur]
                    steps += 1
                tau_eff[s, i, j] = t_eff
    return nu, active, tau_eff


def _fast_accumulate_goods(S, I, n_nodes, n_edges, zone_matrix, active, Q_ship, v, nxt_node, nxt_edge, tau):
    new_goods = np.zeros((S, I, n_edges))
    new_goods_price = np.zeros((S, I, n_edges))
    for s in range(S):
        for j in range(I):
            dst_nodes = zone_matrix[j]
            is_dst = np.zeros(n_nodes, dtype=bool)
            for d in dst_nodes:
                if d >= 0:
                    is_dst[d] = True
            for i in range(I):
                if i == j or not active[s, i, j]:
                    continue
                vol = Q_ship[s, i, j]
                if vol <= 0:
                    continue
                best_val = -np.inf
                best_src = -1
                for src in zone_matrix[i]:
                    if src >= 0 and v[s, j, src] > best_val:
                        best_val = v[s, j, src]
                        best_src = src
                cur = best_src
                steps = 0
                while cur >= 0 and not is_dst[cur] and steps < n_nodes:
                    e = nxt_edge[s, j, cur]
                    nxt = nxt_node[s, j, cur]
                    if e < 0:
                        break
                    new_goods[s, j, e] += vol
                    new_goods_price[s, j, e] += v[s, j, nxt] * vol
                    vol *= tau[s, e]
                    cur = nxt
                    steps += 1
    return new_goods, new_goods_price


class CommuterRoutingBlock:
    def __init__(self):
        self._indptr = None
        self._indices = None
        self._edge_ids = None
        self._zone_matrix = None

    def _prepare_graph(self, net, I):
        if self._indptr is None:
            n = net.n_nodes
            indptr = np.zeros(n + 1, dtype=np.int32)
            indices, edge_ids = [], []
            for u in range(n):
                for v, e in net.adj_out[u]:
                    indices.append(v)
                    edge_ids.append(e)
                indptr[u + 1] = len(indices)
            self._indptr = indptr
            self._indices = np.array(indices, dtype=np.int32)
            self._edge_ids = np.array(edge_ids, dtype=np.int32)
            max_len = max(len(net.zone_to_node[i]) for i in range(I))
            z_mat = np.full((I, max_len), -1, dtype=np.int32)
            for i in range(I):
                z_mat[i, :len(net.zone_to_node[i])] = net.zone_to_node[i]
            self._zone_matrix = z_mat

    def solve(self, state, params):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        self._prepare_graph(net, I)
        T = params.t_link(state.Q_hat, state.I_infra)
        self.dist = np.full((S, I, I), np.inf)
        self.pred_node = np.full((S, I, I, net.n_nodes), -1, dtype=np.int32)
        self.pred_edge = np.full((S, I, I, net.n_nodes), -1, dtype=np.int32)
        self.best_dest = np.full((S, I, I), -1, dtype=np.int32)
        for s in range(S):
            for j in range(I):
                weights = state.w[s, j] * params.d_w * T + state.gamma
                dest_nodes = np.asarray(net.zone_to_node[j], dtype=np.int32)
                for i in range(I):
                    src_nodes = np.asarray(net.zone_to_node[i], dtype=np.int32)
                    d, pn, pe = _fast_dijkstra(net.n_nodes, src_nodes, weights, self._indptr, self._indices, self._edge_ids)
                    self.pred_node[s, j, i] = pn
                    self.pred_edge[s, j, i] = pe
                    best_dest = int(dest_nodes[np.argmin(d[dest_nodes])])
                    self.best_dest[s, j, i] = best_dest
                    self.dist[s, i, j] = d[best_dest]
        state.q_hat_sj = _fast_accumulate(S, I, net.n_nodes, net.n_edges, state.q_pop, self.best_dest, self.pred_node, self.pred_edge, self._zone_matrix)
        state.q_hat_sj = np.nan_to_num(state.q_hat_sj, nan=0.0, posinf=1e8, neginf=0.0)
        state.q_hat_sj = np.clip(state.q_hat_sj, 0.0, 1e8)
        state.mu_commuter = self.dist


class GoodsRoutingBlock:
    def __init__(self):
        self._u_arr = None
        self._v_arr = None
        self._e_arr = None
        self._zone_matrix = None

    def _prepare_graph(self, net, I):
        if self._u_arr is None:
            u_list, v_list, e_list = [], [], []
            for node_idx in range(net.n_nodes):
                for predecessor, edge_idx in net.adj_in[node_idx]:
                    u_list.append(predecessor)
                    v_list.append(node_idx)
                    e_list.append(edge_idx)
            self._u_arr = np.array(u_list, dtype=np.int32)
            self._v_arr = np.array(v_list, dtype=np.int32)
            self._e_arr = np.array(e_list, dtype=np.int32)
            max_len = max(len(net.zone_to_node[i]) for i in range(I))
            z_mat = np.full((I, max_len), -1, dtype=np.int32)
            for i in range(I):
                z_mat[i, :len(net.zone_to_node[i])] = net.zone_to_node[i]
            self._zone_matrix = z_mat

    def solve(self, state, params, hyperparams, t):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        self._prepare_graph(net, I)
        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))
        self.v = np.full((S, I, net.n_nodes), -np.inf)
        self.next_node = np.full((S, I, net.n_nodes), -1, dtype=np.int32)
        self.next_edge = np.full((S, I, net.n_nodes), -1, dtype=np.int32)
        for s in range(S):
            gamma_phi = state.gamma * (params.phi[s] / params.d_w)
            for j in range(I):
                dest_nodes = np.asarray(net.zone_to_node[j], dtype=np.int32)
                p_sj = state.p[s, j]
                v, nxt_node, nxt_edge = _fast_bellman_ford(net.n_nodes, dest_nodes, p_sj, tau[s], gamma_phi, self._u_arr, self._v_arr, self._e_arr)
                self.v[s, j] = v
                self.next_node[s, j] = nxt_node
                self.next_edge[s, j] = nxt_edge
        self.nu_goods, state.active, state.tau_eff = _fast_compute_surplus(S, I, net.n_nodes, self._zone_matrix, self.v, self.next_node, self.next_edge, tau, state.p)
        state.Q_tilde = state.Q_ship * state.tau_eff
        self.accumulate_goods(state, params, hyperparams, t)

    def accumulate_goods(self, state, params, hyperparams, t):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))
        new_goods, new_goods_price = _fast_accumulate_goods(S, I, net.n_nodes, net.n_edges, self._zone_matrix, state.active, state.Q_ship, self.v, self.next_node, self.next_edge, tau)
        theta = 1.0 / (1 + t)
        state.Q_hat_sj_goods = (1 - theta) * state.Q_hat_sj_goods + theta * new_goods
        state.Q_hat_sj_price_goods = (1 - theta) * state.Q_hat_sj_price_goods + theta * new_goods_price
        state.Q_hat_sj_goods = np.nan_to_num(state.Q_hat_sj_goods, nan=0.0, posinf=1e8, neginf=0.0)
        state.Q_hat_sj_price_goods = np.nan_to_num(state.Q_hat_sj_price_goods, nan=0.0, posinf=1e8, neginf=0.0)
        state.Q_hat_sj_goods = np.clip(state.Q_hat_sj_goods, 0.0, 1e8)
        state.Q_hat_sj_price_goods = np.clip(state.Q_hat_sj_price_goods, 0.0, 1e8)


class AnalyticalPrimalBlock:
    def solve(self, state, params):
        S, I = params.n_sectors, params.n_zones
        p_b = state.p[:, np.newaxis, :, np.newaxis]
        r_b = state.r[np.newaxis, :, np.newaxis]
        w_b = state.w[:, np.newaxis, :]
        s_idx = np.arange(S)[:, np.newaxis, np.newaxis]
        i_idx = np.arange(I)[np.newaxis, :, np.newaxis]
        j_idx = np.arange(I)[np.newaxis, np.newaxis, :]
        c, l, f = params.utility_function.invert(p_b, r_b, w_b, s=s_idx, i=i_idx, j=j_idx)
        state.c[:] = c
        state.l_res[:] = l
        state.f[:] = f
        p_sj_b = state.p
        p_vec = state.p[:, np.newaxis, :]
        Y, H, L, Q_inp = params.production_function.invert(p_sj_b, p_vec, state.w, state.r[np.newaxis, :], s=np.arange(S)[:, np.newaxis], j=np.arange(I)[np.newaxis, :], H_sj=0)
        state.Y[:] = Y
        state.H_prod[:] = H
        state.L_prod[:] = L
        state.Q_inter[:] = Q_inp
        U = params.utility_function(state.c, state.l_res, state.f)
        w_b = state.w[:, None, :]
        r_b = state.r[None, :, None]
        p_c_dot = np.einsum('ki,ksij->sij', state.p, state.c)
        V = U + w_b * params.H_bar - r_b * state.l_res - p_c_dot - state.mu_commuter - w_b * state.f
        sigma_arr = params.sigma[:, None, None]
        scaled_V = np.clip(V / sigma_arr, -60.0, 60.0)
        A = params.sigma_tilde * logsumexp(scaled_V, axis=(1, 2))
        log_norm_s = logsumexp(np.clip(A / params.sigma_tilde, -60.0, 60.0))
        q_bar_exp = np.clip(A / params.sigma_tilde - log_norm_s, -50.0, 50.0)
        state.q_bar_s = np.clip(params.q_bar * np.exp(q_bar_exp), 0.0, 1e8)
        ln_sum = logsumexp(scaled_V, axis=(1, 2))[:, None, None]
        state.q_pop = np.clip(state.q_bar_s[:, None, None] * np.exp(np.clip(scaled_V - ln_sum, -50.0, 50.0)), 0.0, 1e8)

        # Fill extended LaTeX variables for the analytical primal / demand layer.
        state.q_ijs[:] = state.q_pop
        state.q_ijsb[:] = state.q_pop[..., None]
        state.q_ijsb_s_k[:] = state.q_pop[..., None, None]
        state.c_ijsb_s[:] = state.c[..., None]
        state.d_ijsb_s_k[:] = state.c[..., None, None]
        state.l_ijsb[:] = state.l_res[..., None]
        state.f_h_ijsb[:] = state.f[..., None]
        state.f_ijsb_s_k[:] = state.f[..., None, None]
        state.H_prod_sb[:] = state.H_prod[..., None]
        state.H_j_p[:] = np.sum(state.H_prod, axis=0)
        state.L_prod_sb[:] = state.L_prod[..., None]
        state.Y_j_p[:] = np.sum(state.Y, axis=0)
        state.w_j_s[:] = state.w
        state.p_i_s_T[:] = state.p
        state.p_i_s_N[:] = state.p
        if hasattr(self, 'v'):
            shape = state.nu_s_ij_n.shape
            state.nu_s_ij_n[:] = np.broadcast_to(self.v[:, None, :, :], shape)
        state.mu_ijsb_n[:] = np.broadcast_to(state.mu_commuter[..., None], state.mu_ijsb_n.shape)
        state.mu_ijsb_s_k_n[:] = np.broadcast_to(state.mu_commuter[..., None, None], state.mu_ijsb_s_k_n.shape)
        state.mu_0js_n[:] = np.broadcast_to(state.mu_commuter[:, None, :], state.mu_0js_n.shape)


class PrimalDualGradientBlock:
    def _excess_goods(self, state, params):
        cons = np.einsum('ksij,sij->ki', state.c, state.q_pop)
        inter = state.Q_inter.sum(axis=0)
        ship_out = state.Q_ship.sum(axis=2) - np.diagonal(state.Q_ship, axis1=1, axis2=2)
        recv_in = state.Q_tilde.sum(axis=1) - np.diagonal(state.Q_tilde, axis1=1, axis2=2)
        return cons + inter + ship_out - state.Y - recv_in

    def _excess_time(self, state, params):
        T_edge = params.t_link(state.Q_hat, state.I_infra)
        commute_time = params.d_w * (state.q_hat_sj * T_edge[None, None, :]).sum(axis=2)
        return state.H_prod + commute_time - state.q_pop.sum(axis=1) * params.H_bar

    def _excess_land(self, state, params):
        prod_land = state.L_prod.sum(axis=0)
        res_land = (state.q_pop * state.l_res).sum(axis=(0, 2))
        return prod_land + res_land - params.L_bar

    def update_primal(self, state, params, hyperparams):
        grad_goods = state.nu_goods - state.p[:, :, None]
        for idx in range(params.n_zones):
            grad_goods[:, idx, idx] = 0.0
        state.Q_ship = np.clip(state.Q_ship + hyperparams.alpha_Q * grad_goods, 0.0, 1e8)
        state.Q_tilde = state.Q_ship * state.tau_eff

    def update_dual(self, state, params, hyperparams):
        ED_goods = self._excess_goods(state, params)
        ED_time = self._excess_time(state, params)
        ED_land = self._excess_land(state, params)
        state.p = np.maximum(0.0, state.p - hyperparams.alpha_lambda * ED_goods)
        state.w = np.maximum(0.0, state.w - hyperparams.alpha_lambda * ED_time)
        state.r = np.maximum(0.0, state.r - hyperparams.alpha_lambda * ED_land)


class LinkCouplingBlock:
    def update_qhat(self, state, params, outer_iter: int):
        # Outer-loop MSA update for link-level aggregate flows.
        # This follows the LaTeX schedule: theta_K = 1 / (K+1) for 0-based K.
        goods_flow = (params.phi[:, None, None] / params.d_w) * state.Q_hat_sj_goods
        Q_hat_induced = goods_flow.sum(axis=(0, 1)) + state.q_hat_sj.sum(axis=(0, 1))
        theta = 1.0 / max(1, outer_iter + 1)
        state.Q_hat = (1 - theta) * state.Q_hat + theta * Q_hat_induced
        state.Q_hat = np.nan_to_num(state.Q_hat, nan=0.0, posinf=1e8, neginf=0.0)
        state.Q_hat = np.clip(state.Q_hat, 0.0, 1e8)

        # Outer-loop MSA smoothing for parking hours and boarding counts.
        parking_demand = np.sum(state.q_pop, axis=(0, 1))
        if state.y_k_P.size == parking_demand.shape[0]:
            state.y_k_P = (1 - theta) * state.y_k_P + theta * parking_demand
        elif state.y_k_P.size >= parking_demand.shape[0]:
            state.y_k_P[:parking_demand.shape[0]] = (1 - theta) * state.y_k_P[:parking_demand.shape[0]] + theta * parking_demand

        boarding_counts = state.Q_hat_sj_goods.sum(axis=(0, 1))
        if state.Q_m_B.ndim == 2 and state.Q_m_B.shape[1] == boarding_counts.shape[0]:
            state.Q_m_B[0, :] = (1 - theta) * state.Q_m_B[0, :] + theta * boarding_counts

    def update_gamma(self, state, params, hyperparams, outer_iter: int):
        dT_dQ = np.asarray(params.t_link.gradient(state.Q_hat, state.I_infra, var='Q'))
        dtau_dQ = np.asarray(params.tau_link.gradient(state.Q_hat, state.I_infra, var='Q'))
        if dtau_dQ.ndim == 1:
            dtau_dQ = np.broadcast_to(dtau_dQ, state.Q_hat_sj_goods.shape)
        dT_dQ = np.nan_to_num(dT_dQ, nan=0.0, posinf=1e8, neginf=0.0)
        dtau_dQ = np.nan_to_num(dtau_dQ, nan=0.0, posinf=1e8, neginf=0.0)
        commute_term = dT_dQ * (params.d_w * (state.w[:, :, None] * state.q_hat_sj).sum(axis=(0, 1)))
        goods_term = (dtau_dQ[:, None, :] * state.Q_hat_sj_price_goods).sum(axis=(0, 1))
        gamma_new = commute_term - goods_term
        gamma_new = np.nan_to_num(gamma_new, nan=0.0, posinf=1e8, neginf=0.0)
        alpha = hyperparams.eta_gamma / max(1, outer_iter + 1)
        state.gamma = np.maximum(0.0, state.gamma + alpha * (gamma_new - state.gamma))

        # Update multimodal boarding and parking shadow values.
        if state.gamma_board.size == state.V_m.shape:
            state.gamma_board[:] = np.maximum(0.0, state.lambda_fleet_m[:, None] * state.V_m)
        if state.gamma_park.size == state.rho_js.sum(axis=0).shape:
            state.gamma_park[:] = np.sum(state.rho_js * params.d_w, axis=0)

    def update_beta(self, state, params, hyperparams, outer_iter: int):
        infra_residual = np.dot(params.network.kappa, state.I_infra) - params.K
        state.beta = np.maximum(0.0, state.beta - hyperparams.eta_beta * infra_residual)


class InfrastructureBlock:
    def _objective(self, state, params):
        U = params.utility_function(state.c, state.l_res, state.f)
        p_c = np.einsum('ki,ksij->sij', state.p, state.c)
        w_b = state.w[:, None, :]
        r_b = state.r[None, :, None]
        benefit = np.sum(state.q_pop * (U - p_c - r_b * state.l_res - w_b * state.f - state.mu_commuter))
        production = np.sum(state.Y)
        cost = np.dot(state.gamma, state.Q_hat)
        budget_term = state.beta * max(0.0, np.dot(params.network.kappa, state.I_infra) - params.K)
        return benefit + production - cost - budget_term

    def update(self, state, params, hyperparams, outer_iter: int):
        net = params.network
        dT_dI = np.asarray(params.t_link.gradient(state.Q_hat, state.I_infra, var='I'))
        dtau_dI = np.asarray(params.tau_link.gradient(state.Q_hat, state.I_infra, var='I'))
        if dtau_dI.ndim == 1:
            dtau_dI = np.broadcast_to(dtau_dI, state.Q_hat_sj_goods.shape)
        infra_marginal = np.sum(dtau_dI[:, None, :] * state.Q_hat_sj_price_goods, axis=(0, 1))
        commute_marginal = dT_dI * (params.d_w * (state.w[:, :, None] * state.q_hat_sj).sum(axis=(0, 1)))
        gamma_I = infra_marginal - commute_marginal
        state.gamma_I = np.nan_to_num(gamma_I, nan=0.0, posinf=1e8, neginf=-1e8)
        candidate = state.I_infra + hyperparams.eta_I * state.gamma_I
        candidate = np.clip(candidate, net.I_min, net.I_max)
        current_obj = self._objective(state, params)
        original = state.I_infra.copy()
        state.I_infra = candidate
        candidate_obj = self._objective(state, params)
        delta = candidate_obj - current_obj
        T = hyperparams.sa_temperature(outer_iter)
        if delta >= 0 or (T > 0 and np.random.rand() < np.exp(np.clip(delta / T, -50.0, 50.0))):
            state.I_infra = candidate
        else:
            state.I_infra = original


class InnerLoopBlock:
    def __init__(self):
        self.commuter = CommuterRoutingBlock()
        self.goods = GoodsRoutingBlock()
        self.primal = AnalyticalPrimalBlock()
        self.gradient = PrimalDualGradientBlock()
        self.logger = ConvergenceLoggerInner()

    def solve(self, state, params, hyperparams, outer_iter):
        _ensure_extended_state(state, params)
        # Inner loop: 1) commuting routing, 2) freight routing, 3) analytical primal forms,
        # 4) primal flow update, 5) dual price update.
        for t in range(hyperparams.T_inner):
            Q_ship_old = state.Q_ship.copy()
            self.commuter.solve(state, params)
            self.goods.solve(state, params, hyperparams, t)
            self.primal.solve(state, params)
            self.gradient.update_primal(state, params, hyperparams)
            self.gradient.update_dual(state, params, hyperparams)
            flow_change = np.linalg.norm(state.Q_ship - Q_ship_old)
            self.logger.log(flow_change, 0.0)
            if flow_change < hyperparams.tol_inner:
                break


class OuterLoopBlock:
    def __init__(self):
        self.link_coupling = LinkCouplingBlock()
        self.infrastructure = InfrastructureBlock()
        self.logger = ConvergenceLoggerOuter()

    def solve(self, state, params, hyperparams):
        for K in range(hyperparams.T_outer):
            p_old = state.p.copy()
            w_old = state.w.copy()
            r_old = state.r.copy()
            beta_old = state.beta
            Q_hat_old = state.Q_hat.copy()
            gamma_old = state.gamma.copy()
            inner = InnerLoopBlock()
            inner.solve(state, params, hyperparams, K)
            self.link_coupling.update_qhat(state, params, K)
            self.link_coupling.update_gamma(state, params, hyperparams, K)
            self.infrastructure.update(state, params, hyperparams, K)
            self.link_coupling.update_beta(state, params, hyperparams, K)
            price_change = np.linalg.norm(state.p - p_old) + np.linalg.norm(state.w - w_old) + np.linalg.norm(state.r - r_old)
            flow_change = np.linalg.norm(state.Q_hat - Q_hat_old)
            congestion_change = np.linalg.norm(state.gamma - gamma_old)
            residual = max(price_change, flow_change, congestion_change, abs(state.beta - beta_old))
            prices = [state.p, state.w, state.r]
            production = state.Y
            aggregate_flows = state.Q_hat
            global_conv = {
                'price_change': price_change,
                'flow_change': flow_change,
                'congestion_change': congestion_change,
                'beta_change': abs(state.beta - beta_old),
            }
            self.logger.log(residual, price_change, prices, production, aggregate_flows, global_conv)
            if residual < hyperparams.tol_outer:
                break
        return state
