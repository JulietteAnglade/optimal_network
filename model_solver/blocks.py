import numpy as np
import heapq
from collections import deque
from .models import ModelParams, ModelState, AlgoParams
from scipy.special import logsumexp
import numba as nb




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
        self.history['global_conv'].append(np.array(global_conv, copy=True))




class ConvergenceLoggerInner:
    def __init__(self):
        self.history = {
            'flow_changes': [],
            'congestion_changes': []
        }

    def log(self, flow_changes, congestion_changes):
        self.history['flow_changes'].append(flow_changes)
        self.history['congestion_changes'].append(congestion_changes)

class ConvergenceVisualization:
    pass


""" class HouseholdBlock:
    def solve(self, state: ModelState, params: ModelParams):
        for s in range(params.n_sectors):
            for i in range(params.n_zones):
                p_i = state.p[:, i]
                r_i = state.r[i]
                for j in range(params.n_zones):
                    w_sj = state.w[s,j]
                    c, l, f = params.utility_function.invert(p_i, r_i, w_sj, s=s, i=i, j=j)
                    state.c[:, s, i, j] = c
                    state.l_res[s, i, j] = l
                    state.f[s, i, j] = f


class ProductionBlock:
    def solve(self, state: ModelState, params: ModelParams, t:int=0):
        for s in range(params.n_sectors):
            for j in range(params.n_zones):
                p_sj = state.p[s, j]
                p_j  = state.p[:, j]
                w_sj = state.w[s, j]
                r_j  = state.r[j]
                
                #Computation of H_sj
                
                h= 0
                Y, H, L, Q_inp = params.production_function.invert(
                    p_sj, p_j, w_sj, r_j, s=s, j=j, H_sj=h
                )
                #a = params.eta(params.eta_H, t)
                state.Y[s, j]            = Y
                state.H_prod[s, j]       = H #dump for H ?
                state.L_prod[s, j]       = L
                state.Q_inter[s, :, j]   = Q_inp     # Q_inter[s, k, j] = good k used by sector s """



class HouseholdBlock:
    def solve(self, state: ModelState, params: ModelParams):
        S, I = params.n_sectors, params.n_zones
        
        # Broadcast inputs to eliminate the S x I x I nested loops
        # state.p: (S_goods, I) -> (S_goods, 1, I, 1)
        p_broadcast = state.p[:, np.newaxis, :, np.newaxis]
        # state.r: (I,) -> (1, I, 1)
        r_broadcast = state.r[np.newaxis, :, np.newaxis]
        # state.w: (S, I) -> (S, 1, I)
        w_broadcast = state.w[:, np.newaxis, :]

        # Create structural index grids for the invert function if required
        s_idx = np.arange(S)[:, np.newaxis, np.newaxis] # (S, 1, 1)
        i_idx = np.arange(I)[np.newaxis, :, np.newaxis] # (1, I, 1)
        j_idx = np.arange(I)[np.newaxis, np.newaxis, :] # (1, 1, I)

        # Compute inversion simultaneously for all sectors and zone pairs
        c, l, f = params.utility_function.invert(
            p_broadcast, r_broadcast, w_broadcast, s=s_idx, i=i_idx, j=j_idx
        )

        # In-place assignment to update the state arrays directly
        state.c[:] = c
        state.l_res[:] = l
        state.f[:] = f


class ProductionBlock:
    def solve(self, state: ModelState, params: ModelParams, t: int = 0):
        S, I = params.n_sectors, params.n_zones
        
        p_sj_b = state.p                  # (S, I)
        p_j_b  = state.p[:, np.newaxis, :] # (S, 1, I) -> Broadcasts across sector dimension
        w_sj_b = state.w                  # (S, I)
        r_j_b  = state.r[np.newaxis, :]   # (1, I)
        
        s_idx = np.arange(S)[:, np.newaxis] # (S, 1)
        j_idx = np.arange(I)[np.newaxis, :] # (1, I)
        
    
        Y, H, L, Q_inp = params.production_function.invert(
            p_sj_b, p_j_b, w_sj_b, r_j_b, s=s_idx, j=j_idx, H_sj=0
        )
        
        # In-place assignment
        state.Y[:] = Y
        state.H_prod[:] = H
        state.L_prod[:] = L
        state.Q_inter[:] = Q_inp

class TransportBlock:
    def __init__(self):
        self.commuters = CommuterSubBlock()
        self.goods = GoodsSubBlock()
        self.flow_aggregator = FlowAggregator()

    def solve(self, state: ModelState, params: ModelParams, hyperparams: AlgoParams, t: int = 0):
        self.commuters.solve(state, params, hyperparams, t)
        self.goods.solve(state, params, hyperparams, t)
        self.flow_aggregator.update_qhat(state, params, hyperparams, t)
        self.flow_aggregator.update_gamma(state, params, hyperparams, t)



# =====================================================================
# NUMBA COMPILED KERNELS (Must be defined outside the class)
# =====================================================================

@nb.njit
def _fast_dijkstra(n_nodes, src_nodes, weights, indptr, indices, edge_ids):
    """C-speed Dijkstra using flat CSR graph arrays."""
    dist = np.full(n_nodes, np.inf)
    pred_node = np.full(n_nodes, -1, dtype=np.int32)
    pred_edge = np.full(n_nodes, -1, dtype=np.int32)
    
    pq = [(0.0, np.int32(u)) for u in src_nodes]
    heapq.heapify(pq)
    
    for u in src_nodes:
        dist[u] = 0.0
        
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            e = edge_ids[i]
            nd = d + weights[e]
            
            if nd < dist[v]:
                dist[v] = nd
                pred_node[v] = u
                pred_edge[v] = e
                heapq.heappush(pq, (nd, v))
                
    return dist, pred_node, pred_edge

@nb.njit
def _fast_accumulate(S, I, n_nodes, n_edges, q_pop, best_dest, pred_node, pred_edge, zone_matrix):
    """C-speed reverse path tracing to accumulate network flows."""
    new_q = np.zeros((S, I, n_edges))
    
    for s in range(S):
        for i in range(I):
            src_nodes = zone_matrix[i]
            
            # Fast O(1) set lookup via boolean array
            is_src = np.zeros(n_nodes, dtype=nb.boolean)
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




import numpy as np
from scipy.special import logsumexp

class CommuterSubBlock:
    def __init__(self):
        # Cache for flattened network structures
        self._indptr = None
        self._indices = None
        self._edge_ids = None
        self._zone_matrix = None

    def _prepare_numba_graph(self, net, I):
        """Converts graph structures to Numba-compatible flat arrays ONCE."""
        if self._indptr is None:
            n = net.n_nodes
            indptr = np.zeros(n + 1, dtype=np.int32)
            indices, edge_ids = [], []
            
            for u in range(n):
                for v, e in net.adj_out[u]:
                    indices.append(v)
                    edge_ids.append(e)
                indptr[u+1] = len(indices)
                
            self._indptr = indptr
            self._indices = np.array(indices, dtype=np.int32)
            self._edge_ids = np.array(edge_ids, dtype=np.int32)

            # Pad zone_to_node into a rectangular 2D array for Numba
            max_len = max(len(net.zone_to_node[i]) for i in range(I))
            z_mat = np.full((I, max_len), -1, dtype=np.int32)
            for i in range(I):
                z_mat[i, :len(net.zone_to_node[i])] = net.zone_to_node[i]
            self._zone_matrix = z_mat

    def shortest_paths(self, state, params):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        self._prepare_numba_graph(net, I)

        T = params.t_link(state.Q_hat, state.I_infra)

        self.dist      = np.full((S, I, I), np.inf)
        self.pred_node = np.full((S, I, I, net.n_nodes), -1, dtype=np.int32)
        self.pred_edge = np.full((S, I, I, net.n_nodes), -1, dtype=np.int32)
        self.best_dest = np.full((S, I, I), -1, dtype=np.int32)

        for s in range(S):
            for j in range(I):
                w_edge = state.w[s, j] * params.d_w * T + state.gamma
                dest_nodes = np.asarray(net.zone_to_node[j], dtype=np.int32)
                
                for i in range(I):
                    src_nodes = np.asarray(net.zone_to_node[i], dtype=np.int32)
                    
                    # Execute compiled C-speed Dijkstra
                    d, pn, pe = _fast_dijkstra(
                        net.n_nodes, src_nodes, w_edge, 
                        self._indptr, self._indices, self._edge_ids
                    )
                    
                    self.pred_node[s, j, i] = pn
                    self.pred_edge[s, j, i] = pe
                    best_dest = dest_nodes[np.argmin(d[dest_nodes])]
                    self.best_dest[s, j, i] = best_dest
                    self.dist[s, i, j] = d[best_dest]

    def compute_allocation(self, state, params):
        # 1. Calcul de l'utilité indirecte global (U shape: S, I, I)
        U = params.utility_function(state.c, state.l_res, state.f)
        
        # 2. Préparation des formes pour le broadcasting (S, I, I)
        w_b = state.w[:, np.newaxis, :]                   # (S, 1, J) -> Salaire à la destination j
        r_b = state.r.ravel()[np.newaxis, :, np.newaxis]  # (1, I, 1) -> Loyer à l'origine i
        
        # 3. Produit scalaire vectorisé des prix et de la consommation
        # Einsum multiplie et somme l'axe des biens (k) : p(k,i) * c(k,s,i,j) -> matrice(s,i,j)
        p_c_dot = np.einsum('ki,ksij->sij', state.p, state.c)
        
        # 4. Calcul de l'allocation V en une seule opération
        V = (U 
             + w_b * params.H_bar 
             - r_b * state.l_res 
             - p_c_dot 
             - self.dist 
             - w_b * state.f)
             
        self.V = np.clip(V, -1e6, 1e6)

        # Vectorized LogSumExp and allocations
        sigma_arr = params.sigma[:, None, None]
        scaled_V = np.clip(self.V / sigma_arr, -60.0, 60.0)
        
        # Calculate A across axes 1 and 2 (i and j)
        A = params.sigma_tilde * logsumexp(scaled_V, axis=(1, 2))
        
        log_norm_s = logsumexp(np.clip(A / params.sigma_tilde, -60.0, 60.0))
        q_bar_exp = np.clip(A / params.sigma_tilde - log_norm_s, -50.0, 50.0)
        state.q_bar_s = np.clip(params.q_bar * np.exp(q_bar_exp), 0.0, 1e6)

        # Calculate q_pop using broadcasting
        ln = logsumexp(scaled_V, axis=(1, 2))[:, None, None]
        state.q_pop = np.clip(
            state.q_bar_s[:, None, None] * np.exp(np.clip(scaled_V - ln, -50.0, 50.0)), 
            0.0, 1e6
        )

    def accumulate_flows(self, state, params, hyperparams, t):
        new_q = _fast_accumulate(
            params.n_sectors, params.n_zones, params.network.n_nodes, params.network.n_edges,
            state.q_pop, self.best_dest, self.pred_node, self.pred_edge, self._zone_matrix
        )
        
        a = hyperparams.alpha_Qhat / (1 + t)
        state.q_hat_sj = (1 - a) * state.q_hat_sj + a * new_q
        state.q_hat_sj = np.nan_to_num(state.q_hat_sj, nan=0.0, posinf=1e4, neginf=0.0)
        state.q_hat_sj = np.clip(state.q_hat_sj, 0.0, 1e4)

    def solve(self, state, params, hyperparams, t):
        self.shortest_paths(state, params)
        self.compute_allocation(state, params)
        self.accumulate_flows(state, params, hyperparams, t)
        
    
    

""" class CommuterSubBlock:
    def _dijkstra(self, net, src, weights):
        n = net.n_nodes
        dist      = np.full(n, np.inf)
        pred_node = np.full(n, -1, dtype=int)
        pred_edge = np.full(n, -1, dtype=int)

        src_nodes = np.asarray(src, dtype=int).ravel()
        pq = []
        for u in src_nodes:
            dist[u] = 0.0
            pq.append((0.0, int(u)))
        heapq.heapify(pq)

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, e in net.adj_out[u]:
                nd = d + weights[e]
                if nd < dist[v]:
                    dist[v] = nd
                    pred_node[v] = u
                    pred_edge[v] = e
                    heapq.heappush(pq, (nd, v))
        return dist, pred_node, pred_edge

    def shortest_paths(self, state, params):
        net = params.network
        S, I = params.n_sectors, params.n_zones

        T = params.t_link(state.Q_hat, state.I_infra)        # (n_edges,)

        self.dist      = np.full((S, I, I), np.inf)
        self.pred_node = np.full((S, I, I, net.n_nodes), -1, dtype=int)
        self.pred_edge = np.full((S, I, I, net.n_nodes), -1, dtype=int)
        self.best_dest = np.full((S, I, I), -1, dtype=int)

        for s in range(S):
            for j in range(I):
                w_edge = state.w[s, j] * params.d_w * T + state.gamma
                for i in range(I):
                    src_nodes = net.zone_to_node[i]
                    dest_nodes = net.zone_to_node[j]
                    d, pn, pe = self._dijkstra(net, src_nodes, w_edge)
                    self.pred_node[s, j, i] = pn
                    self.pred_edge[s, j, i] = pe
                    best_dest = dest_nodes[np.argmin(d[dest_nodes])]
                    self.best_dest[s, j, i] = int(best_dest)
                    self.dist[s, i, j] = d[best_dest]


    def compute_allocation(self, state, params):
        S, I = params.n_sectors, params.n_zones
        V = np.zeros((S, I, I))
        for s in range(S):
            for i in range(I):
                    p_i, r_i = state.p[:, i], state.r[i]
                    for j in range(I):
                        c   = state.c[:, s, i, j]
                        lij = state.l_res[s, i, j]
                        f  = state.f[s, i, j]
                        w_sj = state.w[s, j]
                        U   = params.utility_function(c, lij, f, s=s, i=i, j=j)
                        V[s, i, j] = U + w_sj * params.H_bar - r_i * lij - p_i @ c - self.dist[s, i, j] - w_sj * f
                        V[s, i, j] = np.clip(V[s, i, j], -1e6, 1e6)

            A = np.zeros(S)
            for s in range(S):
                scaled_V = np.clip(V[s] / params.sigma[s], -60.0, 60.0)
                A[s] = params.sigma_tilde * logsumexp(scaled_V)

                log_norm_s = logsumexp(np.clip(A / params.sigma_tilde, -60.0, 60.0))
                q_bar_exp = np.clip(A / params.sigma_tilde - log_norm_s, -50.0, 50.0)
                state.q_bar_s = np.clip(params.q_bar * np.exp(q_bar_exp), 0.0, 1e6)

                for s in range(S):
                    scaled_V = np.clip(V[s] / params.sigma[s], -60.0, 60.0)
                    ln = logsumexp(scaled_V)
                    state.q_pop[s] = np.clip(state.q_bar_s[s] * np.exp(np.clip(scaled_V - ln, -50.0, 50.0)), 0.0, 1e6)

                self.V = V


    def solve(self, state, params, hyperparams, t):
        self.shortest_paths(state, params)
        self.compute_allocation(state, params)
        self.accumulate_flows(state, params, hyperparams, t)
    
    def accumulate_flows(self, state, params, hyperparams, t):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        new_q = np.zeros((S, I, net.n_edges))
        for s in range(S):
            for i in range(I):
                src_nodes = net.zone_to_node[i]
                src_set = set(int(u) for u in np.asarray(src_nodes).ravel())
                for j in range(I):
                    if i == j:
                        continue
                    vol = state.q_pop[s, i, j]
                    if vol <= 0:
                        continue
                    dst = int(self.best_dest[s, j, i])
                    if dst < 0:
                        continue
                    cur = dst
                    while cur not in src_set and cur != -1:
                        e = self.pred_edge[s, j, i, cur]
                        if e < 0:
                            break
                        new_q[s, j, e] += vol
                        cur = self.pred_node[s, j, i, cur]
        a = hyperparams.alpha_Qhat / (1 + t)
        state.q_hat_sj = (
            (1 - a) * state.q_hat_sj
            + a * new_q
        )
        state.q_hat_sj = np.nan_to_num(state.q_hat_sj, nan=0.0, posinf=1e4, neginf=0.0)
        state.q_hat_sj = np.clip(state.q_hat_sj, 0.0, 1e4)
 """



""" class GoodsSubBlock:

    def backward_bellman_ford(self, state, params):
        
        #Use Bellman-Ford algorithm instead of SPFA to guarantee convergence.
        #Finds maximum values (not minimum) for paths ending at each destination.
    
        net = params.network
        S, I = params.n_sectors, params.n_zones

        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))

        self.v         = np.full((S, I, net.n_nodes), -np.inf)
        self.next_node = np.full((S, I, net.n_nodes), -1, dtype=int)
        self.next_edge = np.full((S, I, net.n_nodes), -1, dtype=int)

        for s in range(S):
            for j in range(I):
                dest_nodes = net.zone_to_node[j]
                v = self.v[s, j]
                nxt_node = self.next_node[s, j]
                nxt_edge = self.next_edge[s, j]

                v[dest_nodes] = state.p[s, j]

                # Bellman-Ford: iterate n-1 times through all edges
                for iteration in range(net.n_nodes - 1):
                    updated = False

                    # Process all edges (iterate through adjacency list)
                    for node_idx in range(net.n_nodes):
                        if not np.isfinite(v[node_idx]):
                            continue

                        # For each outgoing edge from node_idx in backward direction
                        # (i.e., incoming edge in forward direction)
                        for predecessor, edge_idx in net.adj_in[node_idx]:
                            cand = tau[s, edge_idx] * v[node_idx] - state.gamma[edge_idx] * params.phi[s] / params.d_w

                            if cand > v[predecessor]:
                                v[predecessor] = cand
                                nxt_node[predecessor] = node_idx
                                nxt_edge[predecessor] = edge_idx
                                updated = True

                    if not updated:
                        # Early termination if no improvements
                        break
                    
    def compute_surplus(self, state, params):
        net = params.network
        S, I = params.n_sectors, params.n_zones

        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))

        self.nu = np.full((S, I, I), -np.inf)
        state.active = np.zeros((S, I, I), dtype=bool)
        state.tau_eff = np.zeros((S, I, I))

        for s in range(S):
            for j in range(I):
                dst_nodes = net.zone_to_node[j]
                for i in range(I):
                    if i == j:
                        continue

                    src_nodes = net.zone_to_node[i]
                    src_vals = self.v[s, j, src_nodes]
                    best_src = int(src_nodes[np.argmax(src_vals)])
                    self.nu[s, i, j] = self.v[s, j, best_src]
                    state.active[s, i, j] = (self.nu[s, i, j] >= state.p[s, i])

                    if self.next_edge[s, j, best_src] < 0:
                        state.tau_eff[s, i, j] = 0.0
                        continue

                    t_eff = 1.0
                    cur = best_src
                    dst_set = set(int(u) for u in np.asarray(dst_nodes).ravel())
                    steps = 0
                    max_steps = net.n_nodes  # Limit path traversal to prevent cycles

                    while cur not in dst_set and cur != -1 and steps < max_steps:
                        e = self.next_edge[s, j, cur]
                        if e < 0:
                            break

                        t_eff *= tau[s, e]
                        cur = self.next_node[s, j, cur]
                        steps += 1

                    state.tau_eff[s, i, j] = t_eff

    
    def update_volumes(self, state, params, hyperparams, t):
        S, I = params.n_sectors, params.n_zones

        a = hyperparams.eta(hyperparams.alpha_Q, t)

        for s in range(S):
            for i in range(I):
                for j in range(I):
                    if i == j:
                        continue

                    grad = self.nu[s, i, j] - state.p[s, i]
                    state.Q_ship[s, i, j] = np.clip(state.Q_ship[s, i, j] + a * grad, 0.0, 1e4)

        state.Q_ship = np.nan_to_num(state.Q_ship, nan=0.0, posinf=1e4, neginf=0.0)
        state.Q_tilde = state.Q_ship * state.tau_eff
    
    def accumulate_flows(self, state, params, hyperparams, t):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        new_goods = np.zeros((S, I, net.n_edges))
        new_goods_price = np.zeros((S, I, net.n_edges))
        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))
        for s in range(S):
            for i in range(I):
                for j in range(I):
                    if i == j or not state.active[s, i, j]:
                        continue
                    vol = state.Q_ship[s, i, j]
                    if vol <= 0:
                        continue
                    src_nodes = net.zone_to_node[i]
                    dst_nodes = net.zone_to_node[j]
                    src_vals = self.v[s, j, src_nodes]
                    best_src = int(src_nodes[np.argmax(src_vals)])
                    cur = best_src
                    dst_set = set(int(u) for u in np.asarray(dst_nodes).ravel())
                    steps = 0
                    max_steps = net.n_nodes
                    while cur not in dst_set and cur != -1 and steps < max_steps:
                        e = self.next_edge[s, j, cur]
                        nxt = self.next_node[s, j, cur]
                        if e < 0:
                            break
                        new_goods[s, j, e] += vol
                        new_goods_price[s, j, e] += (
                            self.v[s, j, nxt] * vol
                        )
                        vol *= tau[s, e]
                        cur = nxt
                        steps += 1
        a = hyperparams.alpha_Qhat / (1 + t)
        state.Q_hat_sj_goods = (
            (1 - a) * state.Q_hat_sj_goods
            + a * new_goods
        )
        state.Q_hat_sj_price_goods = (
            (1 - a) * state.Q_hat_sj_price_goods
            + a * new_goods_price
        )
        state.Q_hat_sj_goods = np.nan_to_num(state.Q_hat_sj_goods, nan=0.0, posinf=1e4, neginf=0.0)
        state.Q_hat_sj_price_goods = np.nan_to_num(state.Q_hat_sj_price_goods, nan=0.0, posinf=1e6, neginf=0.0)
        state.Q_hat_sj_goods = np.clip(state.Q_hat_sj_goods, 0.0, 1e4)
        state.Q_hat_sj_price_goods = np.clip(state.Q_hat_sj_price_goods, 0.0, 1e6) 
    
    
    def solve(self, state, params, hyperparams, t):
        self.backward_bellman_ford(state, params)
        self.compute_surplus(state, params)
        self.update_volumes(state, params, hyperparams, t)
        self.accumulate_flows(state, params, hyperparams, t)"""
        


# =====================================================================
# NUMBA COMPILED KERNELS
# =====================================================================

@nb.njit
def _fast_bellman_ford(n_nodes, dest_nodes, p_sj, tau_s, gamma_phi, u_arr, v_arr, e_arr):
    """C-speed Bellman-Ford using flat edge lists."""
    v = np.full(n_nodes, -np.inf)
    nxt_node = np.full(n_nodes, -1, dtype=np.int32)
    nxt_edge = np.full(n_nodes, -1, dtype=np.int32)

    for d in dest_nodes:
        if d >= 0:
            v[d] = p_sj

    for _ in range(n_nodes - 1):
        updated = False
        for i in range(len(e_arr)):
            pred = u_arr[i]
            node = v_arr[i]
            edge = e_arr[i]

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

@nb.njit
def _fast_compute_surplus(S, I, n_nodes, zone_matrix, v, nxt_node, nxt_edge, tau, p):
    """C-speed path tracing to calculate effective transport costs and surplus."""
    nu = np.full((S, I, I), -np.inf)
    tau_eff = np.zeros((S, I, I))
    active = np.zeros((S, I, I), dtype=nb.boolean)

    for s in range(S):
        for j in range(I):
            dst_nodes = zone_matrix[j]
            is_dst = np.zeros(n_nodes, dtype=nb.boolean)
            for d in dst_nodes:
                if d >= 0: is_dst[d] = True

            for i in range(I):
                if i == j: 
                    continue

                # Find best source node
                best_val = -np.inf
                best_src = -1
                for src in zone_matrix[i]:
                    if src >= 0 and v[s, j, src] > best_val:
                        best_val = v[s, j, src]
                        best_src = src

                if best_src == -1:
                    continue

                nu[s, i, j] = best_val
                active[s, i, j] = (best_val >= p[s, i])

                if nxt_edge[s, j, best_src] < 0:
                    continue

                t_eff = 1.0
                cur = best_src
                steps = 0
                while cur >= 0 and not is_dst[cur] and steps < n_nodes:
                    e = nxt_edge[s, j, cur]
                    if e < 0: break
                    t_eff *= tau[s, e]
                    cur = nxt_node[s, j, cur]
                    steps += 1

                tau_eff[s, i, j] = t_eff
                
    return nu, active, tau_eff

@nb.njit
def _fast_accumulate_goods(S, I, n_nodes, n_edges, zone_matrix, active, Q_ship, v, nxt_node, nxt_edge, tau):
    """C-speed path tracing to accumulate link volumes and prices."""
    new_goods = np.zeros((S, I, n_edges))
    new_goods_price = np.zeros((S, I, n_edges))

    for s in range(S):
        for j in range(I):
            dst_nodes = zone_matrix[j]
            is_dst = np.zeros(n_nodes, dtype=nb.boolean)
            for d in dst_nodes:
                if d >= 0: is_dst[d] = True

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
                    if e < 0: break
                    
                    new_goods[s, j, e] += vol
                    new_goods_price[s, j, e] += v[s, j, nxt] * vol
                    vol *= tau[s, e]
                    cur = nxt
                    steps += 1

    return new_goods, new_goods_price


class GoodsSubBlock:
    def __init__(self):
        self._u_arr = None
        self._v_arr = None
        self._e_arr = None
        self._zone_matrix = None

    def _prepare_numba_graph(self, net, I):
        """Flattens adj_in for Numba iteration ONCE."""
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

    def backward_bellman_ford(self, state, params):
        net = params.network
        S, I = params.n_sectors, params.n_zones
        self._prepare_numba_graph(net, I)

        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))

        self.v         = np.full((S, I, net.n_nodes), -np.inf)
        self.next_node = np.full((S, I, net.n_nodes), -1, dtype=np.int32)
        self.next_edge = np.full((S, I, net.n_nodes), -1, dtype=np.int32)

        for s in range(S):
            gamma_phi = state.gamma * (params.phi[s] / params.d_w)
            for j in range(I):
                dest_nodes = np.asarray(net.zone_to_node[j], dtype=np.int32)
                p_sj = state.p[s, j]
                
                v, nxt_node, nxt_edge = _fast_bellman_ford(
                    net.n_nodes, dest_nodes, p_sj, tau[s], gamma_phi,
                    self._u_arr, self._v_arr, self._e_arr
                )
                
                self.v[s, j] = v
                self.next_node[s, j] = nxt_node
                self.next_edge[s, j] = nxt_edge

    def compute_surplus(self, state, params):
        net = params.network
        S, I = params.n_sectors, params.n_zones

        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))

        self.nu, state.active, state.tau_eff = _fast_compute_surplus(
            S, I, net.n_nodes, self._zone_matrix,
            self.v, self.next_node, self.next_edge, tau, state.p
        )

    def update_volumes(self, state, params, hyperparams, t):
        a = hyperparams.eta(hyperparams.alpha_Q, t)
        
        # Vectorized volume update mapping across S, I, J
        # Broadcast state.p (S, I) to (S, I, 1) to subtract from nu (S, I, J)
        grad = self.nu - state.p[:, :, None]
        
        # Zero out diagonal elements (i == j)
        for idx in range(params.n_zones):
            grad[:, idx, idx] = 0.0

        state.Q_ship = np.clip(state.Q_ship + a * grad, 0.0, 1e4)
        state.Q_ship = np.nan_to_num(state.Q_ship, nan=0.0, posinf=1e4, neginf=0.0)
        state.Q_tilde = state.Q_ship * state.tau_eff

    def accumulate_flows(self, state, params, hyperparams, t):
        net = params.network
        S, I = params.n_sectors, params.n_zones

        tau = np.asarray(params.tau_link(state.Q_hat, state.I_infra))
        if tau.ndim == 1:
            tau = np.broadcast_to(tau, (S, net.n_edges))

        new_goods, new_goods_price = _fast_accumulate_goods(
            S, I, net.n_nodes, net.n_edges, self._zone_matrix, state.active, 
            state.Q_ship, self.v, self.next_node, self.next_edge, tau
        )

        a = hyperparams.alpha_Qhat / (1 + t)
        
        state.Q_hat_sj_goods = (1 - a) * state.Q_hat_sj_goods + a * new_goods
        state.Q_hat_sj_price_goods = (1 - a) * state.Q_hat_sj_price_goods + a * new_goods_price
        
        state.Q_hat_sj_goods = np.nan_to_num(state.Q_hat_sj_goods, nan=0.0, posinf=1e4, neginf=0.0)
        state.Q_hat_sj_price_goods = np.nan_to_num(state.Q_hat_sj_price_goods, nan=0.0, posinf=1e6, neginf=0.0)
        
        state.Q_hat_sj_goods = np.clip(state.Q_hat_sj_goods, 0.0, 1e4)
        state.Q_hat_sj_price_goods = np.clip(state.Q_hat_sj_price_goods, 0.0, 1e6)

    def solve(self, state, params, hyperparams, t):
        self.backward_bellman_ford(state, params)
        self.compute_surplus(state, params)
        self.update_volumes(state, params, hyperparams, t)
        self.accumulate_flows(state, params, hyperparams, t)
    


class FlowAggregator:
    def update_qhat(self, state, params, hyperparams=None, t=0):
        # Qhat_{nn'} = sum_s (phi_s/d_w) Q^{s,goods}_{nn'} + sum_s q^s_{nn'}
        weighted_goods   = (params.phi[:, None, None] / params.d_w) * state.Q_hat_sj_goods
        Q_hat_new = weighted_goods.sum(axis=(0,1)) + state.q_hat_sj.sum(axis=(0,1))
        
        # Apply damping to prevent explosive growth (matching q_hat_s damping)
        if hyperparams is not None:
            a = hyperparams.alpha_Qhat / (1 + t)
            Q_hat_next = (1 - a) * state.Q_hat + a * Q_hat_new
        else:
            Q_hat_next = Q_hat_new
        Q_hat_next = np.nan_to_num(Q_hat_next, nan=0.0, posinf=1e4, neginf=0.0)
        state.Q_hat = np.clip(Q_hat_next, 0.0, 1e4)

    def update_gamma(self, state, params, hyperparams, t):
        # dT/dQ and dtau/dQ at the new Q_hat
        dT_dQ   = np.asarray(params.t_link.gradient(state.Q_hat, state.I_infra, var='Q'))
        dtau_dQ = np.asarray(params.tau_link.gradient(state.Q_hat, state.I_infra, var='Q'))
        if dtau_dQ.ndim == 1:
            dtau_dQ = np.broadcast_to(dtau_dQ, state.Q_hat_sj_goods.shape)

        dT_dQ = np.nan_to_num(dT_dQ, nan=0.0, posinf=1e6, neginf=-1e6)
        dtau_dQ = np.nan_to_num(dtau_dQ, nan=0.0, posinf=1e6, neginf=-1e6)

        commute_term = dT_dQ * (params.d_w * (state.w[:, :, None] * state.q_hat_sj).sum(axis=(0,1)))
        goods_term   = (dtau_dQ[:, None, :] * state.Q_hat_sj_price_goods).sum(axis=(0,1))

        gamma_new = commute_term - goods_term
        gamma_new = np.nan_to_num(gamma_new, nan=0.0, posinf=1e6, neginf=0.0)

        a = hyperparams.eta_gamma/(1+t)  # could use iteration count for decay
        delta = np.clip(a * (gamma_new - state.gamma), -1e2, 1e2)
        state.gamma = np.maximum(0.0, state.gamma + delta)
    


class InfrastructureBlock:
    def update(self, state, params, hyperparams, t):
        net = params.network
        dT_dI   = np.asarray(params.t_link.gradient(state.Q_hat, state.I_infra, var='I'))
        dtau_dI = np.asarray(params.tau_link.gradient(state.Q_hat, state.I_infra, var='I'))
        
        a = hyperparams.eta(hyperparams.eta_I, t)  # could use iteration count for decay

        state.gamma_I = dtau_dI.sum() * state.Q_hat_sj_price_goods.sum(axis=(0,1)) - dT_dI * (params.d_w * (state.w[:, :, None] * state.q_hat_sj).sum(axis=(0,1)))
        state.gamma_I = np.clip(np.nan_to_num(state.gamma_I, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)

        violation = max(0.0, (net.kappa * state.I_infra).sum() - params.K)
        beta_eff = np.clip(state.beta * violation, 0.0, 1e8)
        I_new = state.I_infra + a * (state.gamma_I - beta_eff * net.kappa)
        if np.any(net.I_min > net.I_max):
            raise ValueError("Invalid infrastructure bounds: I_min must be <= I_max for all edges.")
        state.I_infra = np.clip(I_new, net.I_min, net.I_max)
