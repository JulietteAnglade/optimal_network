"""Transportation sub-problem: passenger routing via single-destination
Dijkstra (Section 3.1, eq. 5-8) and freight routing via the
Bellman-Ford-Iceberg dynamic program (Section 3.2, eq. 9-13).

Two routing modes are implemented, selected by `AlgoParams.routing_mode`:

- "discrete" (default): exactly as specified by the document -- heap-based
  Dijkstra and hard-max Bellman-Ford-Iceberg, with all-or-nothing path flow
  assignment (a single shortest path absorbs the household group's / OD
  pair's entire volume).
- "continuous": an entropy-regularized (soft-min / soft-max) relaxation of
  the same recursions, with logit-weighted fractional flow loading instead
  of all-or-nothing assignment. This is a stabilizing extension beyond the
  literal document: smoothing keeps route choice (and hence the FONC
  residuals feeding the primal-dual gradient steps) continuous in the
  background prices instead of jumping discontinuously between competing
  shortest paths, which can damp oscillation in the inner loop. The
  smoothing parameter is `AlgoParams.routing_temperature` (sigma); sigma
  -> 0 recovers the discrete solution.

Both modes are computed by a single-destination sweep per (household
group / OD pair): one Dijkstra / Bellman-Ford call per destination,
returning potentials to every other node at once.
"""
import heapq

import numpy as np

from .algo_params import AlgoParams
from .network import Network
from .params import ModelParams
from .state import ModelState

_BIG = 1e12


# =============================================================================
# Discrete (exact) building blocks
# =============================================================================

def _csr_from_adjacency(adj, n_nodes):
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indices, edge_ids = [], []
    for u in range(n_nodes):
        for v, e in adj[u]:
            indices.append(v)
            edge_ids.append(e)
        indptr[u + 1] = len(indices)
    return indptr, np.array(indices, dtype=np.int64), np.array(edge_ids, dtype=np.int64)


def _dijkstra(n_nodes, src_nodes, weights, indptr, indices, edge_ids):
    """Single-source(s) Dijkstra on the CSR graph (indptr, indices, edge_ids)
    with nonnegative edge `weights`."""
    dist = np.full(n_nodes, np.inf)
    pred_node = np.full(n_nodes, -1, dtype=np.int64)
    pred_edge = np.full(n_nodes, -1, dtype=np.int64)
    pq = [(0.0, int(u)) for u in src_nodes]
    for u in src_nodes:
        dist[u] = 0.0
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for i in range(indptr[u], indptr[u + 1]):
            v = int(indices[i])
            e = int(edge_ids[i])
            w = weights[e]
            if w >= _BIG:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred_node[v] = u
                pred_edge[v] = e
                heapq.heappush(pq, (nd, v))
    return dist, pred_node, pred_edge


def _walk_path(pred_node, pred_edge, start, dest_set, max_steps):
    """Walk from `start` toward the destination along the forward graph
    using pred_node/pred_edge produced by a destination-rooted Dijkstra."""
    cur = start
    steps = 0
    while cur not in dest_set and pred_edge[cur] >= 0 and steps < max_steps:
        e = int(pred_edge[cur])
        nxt = int(pred_node[cur])
        yield cur, nxt, e
        cur = nxt
        steps += 1


class _GraphCache:
    """Caches the CSR adjacency (forward and reversed) for a Network."""

    def __init__(self, net: Network):
        self.net = net
        self.indptr_out, self.indices_out, self.edge_ids_out = _csr_from_adjacency(net.adj_out, net.n_nodes)
        self.indptr_in, self.indices_in, self.edge_ids_in = _csr_from_adjacency(net.adj_in, net.n_nodes)


def _dijkstra_to_destination(net: Network, g: _GraphCache, dest_nodes, weights):
    """mu[n] = shortest-path cost from n to (any node in) dest_nodes along
    *forward* edges, computed by running Dijkstra from dest_nodes on the
    reversed graph. pred_node[n]/pred_edge[n] give the next hop toward the
    destination along the original forward graph."""
    return _dijkstra(net.n_nodes, dest_nodes, weights, g.indptr_in, g.indices_in, g.edge_ids_in)


# =============================================================================
# Continuous (entropy-regularized) building blocks
# =============================================================================

def _soft_relax_to_destination(net: Network, dest_nodes, weights, n_iter, sigma):
    """Soft-min relaxation: v[n] -> sigma * (-log sum_{n'} exp(-(c(n,n')+v[n'])/sigma)),
    iterated as a Bellman-Ford-style sweep (no heap shortcut exists once the
    combine operator is a softmin rather than a hard min). v[dest] = 0 is
    held as the terminal condition, mirroring the discrete case where the
    destination's own distance to itself is exactly 0."""
    dest_set = set(int(d) for d in dest_nodes)
    v = np.full(net.n_nodes, _BIG)
    for d in dest_set:
        v[d] = 0.0
    for _ in range(n_iter):
        v_new = v.copy()
        for u in range(net.n_nodes):
            if u in dest_set:
                continue
            neighbors = net.adj_out[u]
            if not neighbors:
                continue
            logits = []
            for (w_node, e) in neighbors:
                c = weights[e]
                if c >= _BIG or v[w_node] >= _BIG:
                    continue
                logits.append(-(c + v[w_node]) / sigma)
            if not logits:
                continue
            m = max(logits)
            s = sum(np.exp(l - m) for l in logits)
            v_new[u] = -sigma * (m + np.log(s))
        v = v_new
    return v


def _soft_load_flow(net: Network, origin, dest, vol, weights, v, n_iter, sigma, edge_flow_out):
    """Propagate `vol` units of mass from `origin` toward `dest` using
    logit transition probabilities p(u -> w) ~ exp(-(c(u,w)+v[w])/sigma)
    implied by the soft-min value field `v`, accumulating expected edge
    flow into `edge_flow_out` (modified in place). Mass that reaches
    `dest` is absorbed (no further propagation)."""
    if vol <= 0:
        return
    mass = np.zeros(net.n_nodes)
    mass[origin] = vol
    for _ in range(n_iter):
        new_mass = np.zeros(net.n_nodes)
        new_mass[dest] += mass[dest]
        active = False
        for u in range(net.n_nodes):
            m_u = mass[u]
            if u == dest or m_u <= 1e-12:
                continue
            neighbors = net.adj_out[u]
            if not neighbors:
                continue
            logits, edges, targets = [], [], []
            for (w_node, e) in neighbors:
                c = weights[e]
                if c >= _BIG or v[w_node] >= _BIG:
                    continue
                logits.append((v[u] - c - v[w_node]) / sigma)
                edges.append(e)
                targets.append(w_node)
            if not logits:
                continue
            m = max(logits)
            exps = np.exp(np.array(logits) - m)
            probs = exps / exps.sum()
            for p, e, w_node in zip(probs, edges, targets):
                flow = m_u * p
                edge_flow_out[e] += flow
                new_mass[w_node] += flow
                active = True
        mass = new_mass
        if not active:
            break


def _soft_bellman_iceberg(net: Network, dest_nodes, terminal_value, tau_row, gamma_phi, freight_mask, n_iter, sigma):
    """Soft-max relaxation of the Bellman-Iceberg operator (eq. 11):
    v[n] -> sigma * log sum_{n'} exp((tau[e]*v[n'] - gamma_phi[e]) / sigma)."""
    dest_set = set(int(d) for d in dest_nodes)
    v = np.full(net.n_nodes, -np.inf)
    for d in dest_set:
        v[d] = terminal_value
    for _ in range(n_iter):
        v_new = v.copy()
        for u in range(net.n_nodes):
            if u in dest_set:
                continue
            logits = []
            for (w_node, e) in net.adj_out[u]:
                if not freight_mask[e] or not np.isfinite(v[w_node]):
                    continue
                logits.append((tau_row[e] * v[w_node] - gamma_phi[e]) / sigma)
            if not logits:
                continue
            m = max(logits)
            s = sum(np.exp(l - m) for l in logits)
            v_new[u] = sigma * (m + np.log(s))
        v = v_new
    return v


def _soft_load_goods(net: Network, origin, dest, vol, v, tau_row, gamma_phi, freight_mask, sigma, n_iter,
                      flow_out, value_out):
    """Iceberg-aware analogue of _soft_load_flow: mass decays multiplicatively
    by tau[e] each time it crosses an edge (in addition to the logit split)."""
    if vol <= 0:
        return
    mass = np.zeros(net.n_nodes)
    mass[origin] = vol
    for _ in range(n_iter):
        new_mass = np.zeros(net.n_nodes)
        new_mass[dest] += mass[dest]
        active = False
        for u in range(net.n_nodes):
            m_u = mass[u]
            if u == dest or m_u <= 1e-12:
                continue
            logits, edges, targets = [], [], []
            for (w_node, e) in net.adj_out[u]:
                if not freight_mask[e] or not np.isfinite(v[w_node]):
                    continue
                logits.append((tau_row[e] * v[w_node] - gamma_phi[e]) / sigma)
                edges.append(e)
                targets.append(w_node)
            if not logits:
                continue
            m = max(logits)
            exps = np.exp(np.array(logits) - m)
            probs = exps / exps.sum()
            for p, e, w_node in zip(probs, edges, targets):
                flow = m_u * p
                flow_out[e] += flow
                value_out[e] += v[w_node] * flow
                new_mass[w_node] += flow * tau_row[e]
                active = True
        mass = new_mass
        if not active:
            break


# =============================================================================
# Commuter routing (eq. 5-6)
# =============================================================================

class CommuterRoutingBlock:
    def __init__(self):
        self._graph = None

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams):
        net = params.network
        if self._graph is None:
            self._graph = _GraphCache(net)
        g = self._graph
        S, I, Br = params.n_sectors, params.n_zones, params.n_b_r
        n_iter = algo.routing_n_iter or net.n_nodes
        sigma = algo.routing_temperature
        continuous = algo.routing_mode == "continuous"

        T = params.congestion_time.value(state.Qhat_nn, state.I_nn)            # (E,)
        T_board_full = params.boarding_time.value(state.Q_mB_nn, state.V_m_nn)  # (M,E)
        boarding = (T_board_full * net.L_m).sum(axis=0)                        # (E,)

        flow_time_weighted = np.zeros(net.n_edges)
        flow_raw = np.zeros(net.n_edges)
        car_imbalance = np.zeros((S, I, I, Br, I))

        for s in range(S):
            for j in range(I):
                j_node = net.zone_to_node(j)
                core = state.gamma_nn + params.omega_w * state.rho_js[s, j] * (T + boarding)  # (E,)
                for b in range(Br):
                    for i in range(I):
                        i_node = net.zone_to_node(i)
                        h_val = state.h_ijsb[s, i, j, b]
                        fh_val = state.fh_ijsb[s, i, j, b]
                        xi_at = state.xi_ijsb_k[s, i, j, b, :]  # (I,) parking cost per zone

                        weights = core.copy()
                        for e, (u, v) in enumerate(net.edges):
                            if not net.is_car[e]:
                                continue
                            xi_u = xi_at[u] if u < I else 0.0
                            xi_v = xi_at[v] if v < I else 0.0
                            extra = (xi_v - xi_u) * h_val
                            if u == i_node:
                                extra += xi_u * fh_val
                            weights[e] += extra

                        # Numerical safeguard: Dijkstra/the soft relaxation both
                        # assume nonnegative edge costs. rho_j^s (eq. 44) is
                        # deliberately left unprojected and can transiently go
                        # negative off-equilibrium; clipping the *routing* cost
                        # here (not the state variable itself) keeps the
                        # shortest-path solver well-posed without altering the
                        # dual's own update rule.
                        weights = np.maximum(weights, 0.0)

                        if continuous:
                            dist = _soft_relax_to_destination(net, [j_node], weights, n_iter, sigma)
                        else:
                            dist, pred_node, pred_edge = _dijkstra_to_destination(net, g, [j_node], weights)
                        state.mu_ijsb_n[s, i, j, b, :] = dist

                        vol = state.q_ijsb[s, i, j, b]
                        if vol <= 0:
                            continue
                        in_car_at, out_car_at = {}, {}
                        if continuous:
                            group_edge_flow = np.zeros(net.n_edges)
                            _soft_load_flow(net, i_node, j_node, vol, weights, dist, n_iter, sigma, group_edge_flow)
                            for e, f in enumerate(group_edge_flow):
                                if f <= 0:
                                    continue
                                u, w_node = net.edges[e]
                                flow_time_weighted[e] += params.omega_w * state.rho_js[s, j] * f
                                flow_raw[e] += f
                                if net.is_car[e]:
                                    out_car_at[u] = out_car_at.get(u, 0.0) + f
                                    in_car_at[w_node] = in_car_at.get(w_node, 0.0) + f
                        else:
                            for cur, nxt, e in _walk_path(pred_node, pred_edge, i_node, {j_node}, net.n_nodes):
                                flow_time_weighted[e] += params.omega_w * state.rho_js[s, j] * vol
                                flow_raw[e] += vol
                                if net.is_car[e]:
                                    out_car_at[cur] = out_car_at.get(cur, 0.0) + vol
                                    in_car_at[nxt] = in_car_at.get(nxt, 0.0) + vol
                        for node, f in out_car_at.items():
                            if node < I:
                                car_imbalance[s, i, j, b, node] -= f
                        for node, f in in_car_at.items():
                            if node < I:
                                car_imbalance[s, i, j, b, node] += f

        state.flow_time_weighted_nn = flow_time_weighted
        state.flow_raw_nn = flow_raw
        state.car_flow_imbalance_ijsb_k = car_imbalance


# =============================================================================
# Secondary (activity-destination) trip routing (eq. 7)
# =============================================================================

class SecondaryRoutingBlock:
    def __init__(self):
        self._graph = None

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams):
        net = params.network
        if self._graph is None:
            self._graph = _GraphCache(net)
        g = self._graph
        S, I, Br, A = params.n_sectors, params.n_zones, params.n_b_r, params.n_activities
        n_iter = algo.routing_n_iter or net.n_nodes
        sigma = algo.routing_temperature
        continuous = algo.routing_mode == "continuous"

        T = params.congestion_time.value(state.Qhat_nn, state.I_nn)

        for s in range(S):
            for j in range(I):
                core = state.gamma_nn + params.omega_w * state.rho_js[s, j] * T  # eq.7: no boarding term
                for b in range(Br):
                    for i in range(I):
                        i_node = net.zone_to_node(i)
                        xi_at = state.xi_ijsb_k[s, i, j, b, :]
                        for k in range(I):
                            k_node = net.zone_to_node(k)
                            for sp in range(A):
                                f_val = state.f_ijsb_sk[s, i, j, b, k, sp]
                                weights = core.copy()
                                for e, (u, v) in enumerate(net.edges):
                                    if not net.is_car[e]:
                                        continue
                                    xi_u = xi_at[u] if u < I else 0.0
                                    xi_v = xi_at[v] if v < I else 0.0
                                    weights[e] += (xi_v - xi_u) * f_val
                                weights = np.maximum(weights, 0.0)  # see CommuterRoutingBlock note

                                if continuous:
                                    dist = _soft_relax_to_destination(net, [k_node], weights, n_iter, sigma)
                                else:
                                    dist, pred_node, pred_edge = _dijkstra_to_destination(net, g, [k_node], weights)
                                state.mu_ijsb_sk_n[s, i, j, b, k, sp, :] = dist

                                vol = state.q_ijsb_sk[s, i, j, b, k, sp]
                                if vol <= 0:
                                    continue
                                if continuous:
                                    group_edge_flow = np.zeros(net.n_edges)
                                    _soft_load_flow(net, i_node, k_node, vol, weights, dist, n_iter, sigma, group_edge_flow)
                                    for e, f in enumerate(group_edge_flow):
                                        if f > 0:
                                            state.flow_time_weighted_nn[e] += params.omega_w * state.rho_js[s, j] * f
                                            state.flow_raw_nn[e] += f
                                else:
                                    for cur, nxt, e in _walk_path(pred_node, pred_edge, i_node, {k_node}, net.n_nodes):
                                        state.flow_time_weighted_nn[e] += params.omega_w * state.rho_js[s, j] * vol
                                        state.flow_raw_nn[e] += vol


# =============================================================================
# Peripheral commuter routing (eq. 8)
# =============================================================================

class PeripheralRoutingBlock:
    def __init__(self):
        self._graph = None

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams):
        net = params.network
        if self._graph is None:
            self._graph = _GraphCache(net)
        g = self._graph
        S, I = params.n_sectors, params.n_zones
        n_iter = algo.routing_n_iter or net.n_nodes
        sigma = algo.routing_temperature
        continuous = algo.routing_mode == "continuous"

        for s in range(S):
            for j in range(I):
                j_node = net.zone_to_node(j)
                h_val = state.h0_js[s, j]
                weights = state.gamma_nn.copy()
                for e, (u, v) in enumerate(net.edges):
                    if not net.is_car[e]:
                        continue
                    gp_u = state.gamma_park_k[u] if u < I else 0.0
                    gp_v = state.gamma_park_k[v] if v < I else 0.0
                    weights[e] += h_val * (gp_v - gp_u)
                weights = np.maximum(weights, 0.0)  # see CommuterRoutingBlock note

                if continuous:
                    dist = _soft_relax_to_destination(net, [j_node], weights, n_iter, sigma)
                else:
                    dist, pred_node, pred_edge = _dijkstra_to_destination(net, g, [j_node], weights)
                state.mu_0js_n[s, j, :] = dist

                vol = state.q0_js[s, j]
                if vol > 0:
                    if continuous:
                        group_edge_flow = np.zeros(net.n_edges)
                        _soft_load_flow(net, net.entry_node, j_node, vol, weights, dist, n_iter, sigma, group_edge_flow)
                        state.flow_raw_nn += group_edge_flow
                    else:
                        for cur, nxt, e in _walk_path(pred_node, pred_edge, net.entry_node, {j_node}, net.n_nodes):
                            state.flow_raw_nn[e] += vol


# =============================================================================
# Freight routing: Bellman-Ford-Iceberg (eq. 9-13)
# =============================================================================

class FreightRoutingBlock:
    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        n_iter = algo.routing_n_iter or net.n_nodes
        sigma = algo.routing_temperature
        continuous = algo.routing_mode == "continuous"
        tau = params.iceberg_tau.value(state.Qhat_nn, state.I_nn)  # (S, E)

        flow_goods_nn_s = np.zeros((S, net.n_edges))
        flow_goods_value_nn_s = np.zeros((S, net.n_edges))

        for s in range(S):
            gamma_phi = state.gamma_nn * params.phi[s] / params.omega_w  # (E,)
            for j in range(I):
                j_node = net.zone_to_node(j)
                terminal_value = state.p_sT[s, j]

                if continuous:
                    v = _soft_bellman_iceberg(net, [j_node], terminal_value, tau[s], gamma_phi,
                                               net.freight_allowed, n_iter, sigma)
                else:
                    v, nxt_node, nxt_edge = _hard_bellman_iceberg(net, [j_node], terminal_value, tau[s], gamma_phi,
                                                                   net.freight_allowed, n_iter)
                state.nu_s_ij_n[s, :, j, :] = v[None, :]

                for i in range(I):
                    i_node = net.zone_to_node(i)
                    vol = state.Q_ij_s[s, i, j]
                    if vol <= 0 or i == j:
                        continue
                    if continuous:
                        _soft_load_goods(net, i_node, j_node, vol, v, tau[s], gamma_phi, net.freight_allowed,
                                          sigma, n_iter, flow_goods_nn_s[s], flow_goods_value_nn_s[s])
                    else:
                        cur = i_node
                        steps = 0
                        while cur != j_node and nxt_edge[cur] >= 0 and steps < net.n_nodes:
                            e = int(nxt_edge[cur])
                            nxt = int(nxt_node[cur])
                            flow_goods_nn_s[s, e] += vol
                            flow_goods_value_nn_s[s, e] += v[nxt] * vol
                            vol *= tau[s, e]
                            cur = nxt
                            steps += 1

        state.flow_goods_nn_s = flow_goods_nn_s
        state.flow_goods_value_nn_s = flow_goods_value_nn_s


def _hard_bellman_iceberg(net: Network, dest_nodes, terminal_value, tau_row, gamma_phi, freight_mask, n_iter):
    """Exact Bellman-Ford-Iceberg recursion (eq. 9-13): v_x^0 = b_x,
    v_n^{iter} = max_{n'} (tau[e] * v_{n'}^{iter-1} - gamma_phi[e])."""
    dest_set = set(int(d) for d in dest_nodes)
    v = np.full(net.n_nodes, -np.inf)
    for d in dest_set:
        v[d] = terminal_value
    nxt_node = np.full(net.n_nodes, -1, dtype=np.int64)
    nxt_edge = np.full(net.n_nodes, -1, dtype=np.int64)
    for _ in range(n_iter - 1):
        updated = False
        for u in range(net.n_nodes):
            for (w_node, e) in net.adj_out[u]:
                if not freight_mask[e] or not np.isfinite(v[w_node]):
                    continue
                cand = tau_row[e] * v[w_node] - gamma_phi[e]
                if cand > v[u]:
                    v[u] = cand
                    nxt_node[u] = w_node
                    nxt_edge[u] = e
                    updated = True
        if not updated:
            break
    return v, nxt_node, nxt_edge
