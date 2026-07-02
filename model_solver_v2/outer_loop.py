"""Three-level loop nesting for Section 5.1 steps 2(a), 3-6.

- Outer loop (OuterLoopBlock): structural-only updates -- eq. 62-70's four
  Jacobi-style outer structural updates (infrastructure with simulated
  annealing + multi-start, floor space, parking-hours supply, transit
  vehicle flow + its promoted fleet/conservation duals), the congestion-fee
  refresh (eq. 71-73, itself MSA-smoothed -- see CongestionFeeBlock), and
  the capital-budget shadow price (eq. 74). No MSA fixed-point logic lives
  here directly.
- Intermediate loop (IntermediateLoopBlock): the eq. 61 MSA fixed point on
  the network-loading part of Z (Q_hat_{nn'}, y_k^P, Q_{nn'}^{m,B}),
  obtained by repeating inner-loop solve + MSA-damp until Z stabilizes
  (tol_msa) or T_msa rounds elapse, with the outer-structural state O^{(K)}
  held fixed throughout -- i.e. this *is* step 2(b) plus step 3, iterated
  to its own convergence before the outer loop's structural updates run.
- Inner loop (InnerLoopBlock, inner_loop.py): unchanged -- routing ->
  freight routing -> cascade -> analytical primal -> gradient ascent/descent.

Every quantity here that isn't given a fully explicit closed form in the
document (the "induced" flow aggregations feeding the MSA step, the
per-group parking-hours demand y_k^{P,ijsb} feeding eq. 66/73, and the
objective used to accept/reject simulated-annealing infrastructure moves)
is approximated from the routing flow caches populated in routing.py,
with the approximation documented inline at each site.

Per step 4's own instruction, sub-steps (a)-(d) are mutually Jacobi-style:
each reads only the start-of-cycle (K)-level value of every *other*
outer-structural variable. OuterLoopBlock therefore snapshots the full
outer-structural state once before computing any of (a)-(d).
"""
import numpy as np

from .algo_params import AlgoParams
from .convergence import argmax_residual
from .inner_loop import InnerLoopBlock
from .params import ModelParams
from .state import ModelState

_EPS = 1e-8
_Y_MIN = 1.0  # small positive floor for Y_k^p, see ParkingSupplyBlock
_V_MIN = 0.1  # small positive floor for V_m_nn on transit edges, see TransitFleetBlock


class MSABlock:
    """Eq. 61: theta_K = 1/K damping of the macroscopic coupling vector Z."""

    def solve(self, state: ModelState, params: ModelParams, K: int):
        theta = 1.0 / max(1, K)
        net = params.network

        goods_induced = (params.phi[:, None] / params.omega_w) * state.flow_goods_nn_s
        Q_hat_induced = state.flow_raw_nn + goods_induced.sum(axis=0)
        state.Qhat_nn = (1 - theta) * state.Qhat_nn + theta * Q_hat_induced

        # y_k^P induced: total arrivals at zone k needing to park -- commuters
        # whose workplace is k, plus secondary-activity trips destined at k
        # (the document does not give this aggregation a closed form; this
        # mirrors the same "arrivals at k" logic used for gamma_park_k below).
        commute_at_k = state.q_ijsb.sum(axis=(0, 1, 3))  # sum s,i,b over q_ijsb[s,i,k,b] -> (I,)
        secondary_at_k = state.q_ijsb_sk.sum(axis=(0, 1, 2, 3, 5))  # sum s,i,j,b,sp -> (I,) indexed by k
        yP_induced = commute_at_k + secondary_at_k
        state.yP_k = (1 - theta) * state.yP_k + theta * yP_induced

        # Q_{nn'}^{m,B}: since a transit-mode edge only ever carries mode-m
        # ridership, the aggregate edge flow restricted to L^m is exactly
        # the boarding count for that mode.
        Q_mB_induced = state.flow_raw_nn[None, :] * net.L_m
        state.Q_mB_nn = (1 - theta) * state.Q_mB_nn + theta * Q_mB_induced


class IntermediateLoopBlock:
    """Steps 2(b) + 3, iterated to their own fixed point: repeatedly solve
    the inner loop and MSA-damp Q_hat_{nn'}/y_k^P/Q_{nn'}^{m,B} (the
    network-loading part of Z), with the outer-structural state O^{(K)}
    held fixed throughout, until Z stops moving (tol_msa) or T_msa rounds
    have run. Its own damping counter m resets every time it is entered
    (it is a fixed point *within* a given outer cycle, not across cycles)."""

    def __init__(self):
        self.inner = InnerLoopBlock()
        self.msa = MSABlock()

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams):
        history = []
        for m in range(1, algo.T_msa + 1):
            snapshot = (state.Qhat_nn.copy(), state.yP_k.copy(), state.Q_mB_nn.copy())
            inner_residuals = self.inner.solve(state, params, algo)
            self.msa.solve(state, params, m)
            residual = argmax_residual([
                ('Qhat_nn', state.Qhat_nn - snapshot[0]),
                ('yP_k', state.yP_k - snapshot[1]),
                ('Q_mB_nn', state.Q_mB_nn - snapshot[2]),
            ])
            history.append({'m': m, 'residual': residual, 'inner_iters': len(inner_residuals)})
            if residual['value'] < algo.tol_msa:
                break
        return history


class InfrastructureBlock:
    """Eq. 62-63: projected gradient ascent regulated by simulated
    annealing + periodic multi-start, for the combinatorial network
    non-convexity induced by I_{nn'}. The stationarity condition being
    approximated (the reference FONC for this block) is

        gamma^I_{nn'} <= beta * kappa_{nn'}
                         + alpha^infra * (r^infra_n + r^infra_n') / 2
                         perp  I_{nn'} >= I_min_{nn'}

    i.e. gamma_I (the congestion/freight/fleet benefit computed below) is
    weighed against *two* cost terms: the capital-budget shadow price
    beta times the edge's unit cost kappa_{nn'}, and the land-opportunity
    cost (alpha^infra times the average land price r^infra at the edge's
    two endpoints, mirroring eq. 50's own use of alpha^infra). Both must
    enter the ascent step directly -- previously only the benefit side
    was used, which is why beta had no way to discourage further
    infrastructure spend (see the l_bar-collapse investigation)."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, K: int):
        net = params.network
        dT2 = params.congestion_time.grad2(state.Qhat_nn, state.I_nn)         # (E,)
        dtau2 = params.iceberg_tau.grad2(state.Qhat_nn, state.I_nn)           # (S, E)

        fleet_term = (net.L_m * state.lambda_fleet_m[:, None] * state.V_m_nn).sum(axis=0) * dT2
        gamma_I = (-state.flow_time_weighted_nn * dT2
                   + (dtau2 * state.flow_goods_value_nn_s).sum(axis=0)
                   - fleet_term)

        r_infra_nodes = np.zeros(net.n_nodes)
        r_infra_nodes[:params.n_zones] = state.r_infra  # 0 at the boundary node (no land market there)
        land_cost = np.zeros(net.n_edges)
        for e, (u, v) in enumerate(net.edges):
            land_cost[e] = params.alpha_infra * (r_infra_nodes[u] + r_infra_nodes[v]) / 2.0
        capital_cost = state.beta * net.kappa

        gamma_I_net = gamma_I - capital_cost - land_cost
        I_cand = np.clip(state.I_nn + algo.eta(algo.alpha_I, K) * gamma_I_net, net.I_min, net.I_max)

        current_obj = self._objective(state, params)
        I_old = state.I_nn.copy()
        state.I_nn = I_cand
        candidate_obj = self._objective(state, params)
        delta = candidate_obj - current_obj

        T = state.sa_temperature
        accept = delta >= 0 or (T > 0 and np.random.rand() < np.exp(np.clip(delta / T, -50.0, 50.0)))
        if accept:
            if candidate_obj > state.sa_best_objective:
                state.sa_best_objective = candidate_obj
                state.sa_best_I = I_cand.copy()
        else:
            state.I_nn = I_old

        if algo.sa_multistart_every > 0 and K % algo.sa_multistart_every == 0:
            best_obj, best_I = state.sa_best_objective, state.sa_best_I.copy()
            for _ in range(algo.sa_multistart_count):
                trial = np.random.uniform(net.I_min, net.I_max)
                state.I_nn = trial
                obj = self._objective(state, params)
                if obj > best_obj:
                    best_obj, best_I = obj, trial.copy()
            state.I_nn = best_I
            state.sa_best_objective, state.sa_best_I = best_obj, best_I

        state.sa_temperature *= algo.sa_cooling

    def _objective(self, state: ModelState, params: ModelParams) -> float:
        """Welfare proxy used only to accept/reject simulated-annealing
        moves on I_{nn'} (see class docstring): net of transport and
        capital costs. Re-deriving the exact primal Lagrangian value at
        every candidate move is out of scope here; this reduced proxy
        captures the part of welfare infrastructure changes act on most
        directly (congestion relief net of build cost). The capital cost
        is beta-weighted (plus the land-opportunity term) to stay
        consistent with the gradient step above -- both should be pricing
        I_{nn'} against the same shadow costs."""
        net = params.network
        transport_cost = float(np.dot(state.gamma_nn, state.Qhat_nn))
        r_infra_nodes = np.zeros(net.n_nodes)
        r_infra_nodes[:params.n_zones] = state.r_infra
        land_cost = np.array([
            params.alpha_infra * (r_infra_nodes[u] + r_infra_nodes[v]) / 2.0
            for u, v in net.edges
        ])
        capital_cost = float(np.dot(state.beta * net.kappa + land_cost, state.I_nn))
        return -transport_cost - capital_cost


class FloorSpaceBlock:
    """Eq. 64: plain projected gradient ascent, shared across the three
    building-type families (residential / non-residential / parking)."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, K: int):
        a = algo.eta(algo.alpha_l, K)
        state.l_bar_res = np.maximum(0.0, state.l_bar_res + a * (
            state.r_res - params.alpha_floor_res * state.r_bar_res - state.beta * params.kappa_floor_res))
        state.l_bar_nonres = np.maximum(0.0, state.l_bar_nonres + a * (
            state.r_nonres - params.alpha_floor_nonres * state.r_bar_nonres - state.beta * params.kappa_floor_nonres))
        state.l_bar_park = np.maximum(0.0, state.l_bar_park + a * (
            state.r_park - params.alpha_floor_park * state.r_bar_park - state.beta * params.kappa_floor_park))


class ParkingSupplyBlock:
    """Eq. 65-66: plain projected gradient ascent on Y_k^p (no annealing).
    Mind the sign convention noted in the document: -alpha_Y * dL/dY_k^p."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, K: int):
        S_k = state.Y_pk.sum(axis=1)  # (I,)
        dT_P2 = params.parking_search_time.grad2(state.yP_k, S_k)  # (I,)

        # sum_{ijsb} rho_j^s * omega_w * y_k^{P,ijsb,t*}: the document does
        # not track y_k^{P,ijsb} as its own state variable; approximated
        # (as in MSABlock) by the rho-weighted volume of trips arriving at k.
        commute_at_k = np.einsum('sk,sk->k', state.q_ijsb.sum(axis=(1, 3)), state.rho_js)
        secondary_by_sjk = state.q_ijsb_sk.sum(axis=(1, 3, 5))  # sum i,b,sp -> (S,J,I[k])
        secondary_at_k = np.einsum('sjk,sj->k', secondary_by_sjk, state.rho_js)
        weighted_demand = params.omega_w * (commute_at_k + secondary_at_k)

        # A small positive floor (rather than 0) keeps aggregate parking
        # supply S_k = sum_p Y_k^p from hitting exactly zero: T^P_k ~ 1/S_k
        # (eq. 28) would otherwise blow up without bound the moment supply
        # is driven to its corner, which is a numerical singularity of the
        # 1/S functional form rather than a meaningful economic outcome
        # (parking supply near zero should mean a very high, not infinite,
        # search cost).
        Y_new = np.maximum(_Y_MIN, state.Y_pk - algo.eta(algo.alpha_Y, K) * (
            state.pi_jp + weighted_demand[:, None] * dT_P2[:, None]))
        state.Y_pk = Y_new


class TransitFleetBlock:
    """Eq. 67-70: transit vehicle flow V^m_{nn'} and its promoted fleet
    (lambda^fleet_m) / conservation (lambda^{v_cons}_{m,n}) duals."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, K: int):
        net = params.network
        M = params.n_modes_transit
        T_val = params.congestion_time.value(state.Qhat_nn, state.I_nn)              # (E,)
        T_board = params.boarding_time.value(state.Q_mB_nn, state.V_m_nn)            # (M,E)
        dT_board2 = params.boarding_time.grad2(state.Q_mB_nn, state.V_m_nn)          # (M,E)
        dtW = params.waiting_time.grad(state.V_m_nn)                                 # (M,E)

        lambda_stat_u = np.zeros((M, net.n_edges))
        lambda_stat_v = np.zeros((M, net.n_edges))
        for e, (u, v) in enumerate(net.edges):
            lambda_stat_u[:, e] = state.lambda_stat_mn[:, u]
            lambda_stat_v[:, e] = state.lambda_stat_mn[:, v]

        # sum_{ijsb} rho_j^s * omega_w * q_{nn'}^{ijsb} (commuter-only) and
        # the analogous Q^{m,B}-weighted waiting-time term: approximated by
        # the cached rho-weighted flow (see module docstring) since transit
        # edges only ever carry that mode's own ridership.
        ridership_term = state.flow_time_weighted_nn[None, :] * net.L_m

        Psi = (lambda_stat_u + lambda_stat_v
               + state.lambda_fleet_m[:, None] * (T_val[None, :] + T_board + state.V_m_nn * dT_board2)
               + ridership_term * (dT_board2 + dtW))

        lambda_vcons_u = np.zeros((M, net.n_edges))
        lambda_vcons_v = np.zeros((M, net.n_edges))
        for e, (u, v) in enumerate(net.edges):
            lambda_vcons_u[:, e] = state.lambda_vcons_mn[:, u]
            lambda_vcons_v[:, e] = state.lambda_vcons_mn[:, v]

        # A small positive floor on *transit* edges (rather than 0) keeps
        # V_m_nn from hitting exactly zero there: boarding_time.grad2 and
        # waiting_time.grad both contain a 1/V_m term (marginal returns to
        # added vehicle frequency, singular at V_m = 0 by construction),
        # which the generic 1e-8 floor inside those primitives turns into a
        # ~1e8-1e9 derivative feeding straight into Psi -- the same failure
        # pattern as Y_k^p hitting its own zero corner (see ParkingSupplyBlock).
        # Non-transit edges are still forced to exactly 0 via the L_m mask.
        V_candidate = np.maximum(_V_MIN, state.V_m_nn + algo.eta(algo.alpha_V, K) * (
            state.pi_nn_m + lambda_vcons_v - lambda_vcons_u - Psi))
        V_new = np.where(net.L_m, V_candidate, 0.0)

        a_lambda = algo.eta(algo.alpha_lambda, K)
        fleet_usage = (state.V_m_nn * (T_val[None, :] + T_board) * net.L_m).sum(axis=1)  # (M,)
        lambda_fleet_new = np.maximum(0.0, state.lambda_fleet_m - a_lambda * (
            params.V_bar_m - fleet_usage))

        V_out = np.zeros((M, net.n_nodes))
        V_in = np.zeros((M, net.n_nodes))
        for e, (u, v) in enumerate(net.edges):
            V_out[:, u] += state.V_m_nn[:, e]
            V_in[:, v] += state.V_m_nn[:, e]
        lambda_vcons_new = state.lambda_vcons_mn - a_lambda * (V_in - V_out)  # no projection, per eq.70

        state.V_m_nn = V_new
        state.lambda_fleet_m = lambda_fleet_new
        state.lambda_vcons_mn = lambda_vcons_new


class CongestionFeeBlock:
    """Eq. 71-73: gamma_{nn'}, gamma^{board}_{m,nn'}, gamma^{park}_k,
    evaluated with inner-loop quantities at t* and outer-structural
    quantities (lambda^fleet, V^m) at the just-produced (K+1). These three
    are also part of Z (Section 2.2's macroscopic coupling vector) and the
    document has them MSA-smoothed too ("smooth pricing matrices using the
    MSA protocol before starting outer cycle K+1") -- unlike Q_hat/y^P/Q^{m,B}
    (IntermediateLoopBlock), this smoothing must happen here, after the
    outer structural updates, since it needs their just-produced (K+1)
    values; theta_K uses the outer cycle counter K."""

    def solve(self, state: ModelState, params: ModelParams, K: int):
        theta = 1.0 / max(1, K)
        net = params.network
        dT1 = params.congestion_time.grad1(state.Qhat_nn, state.I_nn)
        dtau1 = params.iceberg_tau.grad1(state.Qhat_nn, state.I_nn)

        fleet_term = (net.L_m * state.lambda_fleet_m[:, None] * state.V_m_nn).sum(axis=0) * dT1
        gamma_nn_induced = np.maximum(0.0, (
            state.flow_time_weighted_nn * dT1
            - (dtau1 * state.flow_goods_value_nn_s).sum(axis=0)
            + fleet_term))
        state.gamma_nn = (1 - theta) * state.gamma_nn + theta * gamma_nn_induced

        dT_board1 = params.boarding_time.grad1(state.Q_mB_nn, state.V_m_nn)  # (M,E)
        ridership_term = state.flow_time_weighted_nn[None, :] * net.L_m
        gamma_board_induced = np.maximum(0.0, (
            state.lambda_fleet_m[:, None] * state.V_m_nn + ridership_term) * dT_board1)
        state.gamma_board_m_nn = (1 - theta) * state.gamma_board_m_nn + theta * gamma_board_induced

        S_k = state.Y_pk.sum(axis=1)
        dT_P1 = params.parking_search_time.grad1(state.yP_k, S_k)
        commute_at_k = np.einsum('sk,sk->k', state.q_ijsb.sum(axis=(1, 3)), state.rho_js)
        secondary_by_sjk = state.q_ijsb_sk.sum(axis=(1, 3, 5))
        secondary_at_k = np.einsum('sjk,sj->k', secondary_by_sjk, state.rho_js)
        gamma_park_induced = np.maximum(0.0, params.omega_w * (commute_at_k + secondary_at_k) * dT_P1)
        state.gamma_park_k = (1 - theta) * state.gamma_park_k + theta * gamma_park_induced


class BetaBlock:
    """Eq. 74: projected gradient descent on the capital-budget shadow price."""

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, K: int):
        infra_cost = float(np.dot(params.network.kappa, state.I_nn))
        floor_cost = (float((state.l_bar_res * params.kappa_floor_res).sum())
                      + float((state.l_bar_nonres * params.kappa_floor_nonres).sum())
                      + float((state.l_bar_park * params.kappa_floor_park).sum()))
        state.beta = max(0.0, state.beta - algo.eta(algo.alpha_beta, K) * (params.K - infra_cost - floor_cost))


class OuterLoopBlock:
    def __init__(self):
        self.intermediate = IntermediateLoopBlock()
        self.infra = InfrastructureBlock()
        self.floor_space = FloorSpaceBlock()
        self.parking_supply = ParkingSupplyBlock()
        self.transit_fleet = TransitFleetBlock()
        self.congestion_fee = CongestionFeeBlock()
        self.beta_block = BetaBlock()
        self.history = []

    def solve(self, state: ModelState, params: ModelParams, algo: AlgoParams, progress_callback=None):
        """progress_callback, if given, is called with this cycle's history
        entry (dict) right after it's recorded -- e.g. `lambda entry:
        print(entry, flush=True)` to monitor convergence live instead of
        only inspecting self.history after solve() returns."""
        for K in range(1, algo.T_outer + 1):
            snapshot = self._snapshot(state)

            # step 2(a): background network cost functions are evaluated
            # on-the-fly from state.Qhat_nn / state.I_nn wherever needed
            # (routing.py, inner_loop.py) -- no separate cache step required.

            # step 2(b) + step 3: inner loop + MSA on Q_hat/y^P/Q^{m,B},
            # iterated to their own fixed point, with O^{(K)} held fixed
            # throughout.
            msa_history = self.intermediate.solve(state, params, algo)

            # step 4(a)-(d): Jacobi-style outer structural updates, each
            # using the start-of-cycle (K)-level value of every other
            # outer-structural variable (already true here since none of
            # these four blocks reads another's *new* output).
            self.infra.solve(state, params, algo, K)
            self.floor_space.solve(state, params, algo, K)
            self.parking_supply.solve(state, params, algo, K)
            self.transit_fleet.solve(state, params, algo, K)

            # step 4(e): congestion fees (also MSA-smoothed, see
            # CongestionFeeBlock), using the just-produced (K+1) fleet
            # dual / vehicle flow
            self.congestion_fee.solve(state, params, K)

            # step 4(f): capital-budget shadow price
            self.beta_block.solve(state, params, algo, K)

            residual = self._residual(state, snapshot)
            inner_iters = sum(entry['inner_iters'] for entry in msa_history)
            entry = {
                'K': K, 'residual': residual,
                'msa_iters': len(msa_history), 'inner_iters': inner_iters,
            }
            self.history.append(entry)
            if progress_callback is not None:
                progress_callback(entry)
            if residual['value'] < algo.tol_outer:
                break
        return state

    @staticmethod
    def _snapshot(state: ModelState):
        return (state.Qhat_nn.copy(), state.I_nn.copy(), state.l_bar_res.copy(),
                state.Y_pk.copy(), state.V_m_nn.copy(), state.beta)

    @staticmethod
    def _residual(state: ModelState, snapshot):
        """Returns {'value','variable','index'} identifying which named
        outer-structural quantity (and which entry within it) is
        responsible for this cycle's max-abs-change."""
        Qhat_old, I_old, l_old, Y_old, V_old, beta_old = snapshot
        return argmax_residual([
            ('Qhat_nn', state.Qhat_nn - Qhat_old),
            ('I_nn', state.I_nn - I_old),
            ('l_bar_res', state.l_bar_res - l_old),
            ('Y_pk', state.Y_pk - Y_old),
            ('V_m_nn', state.V_m_nn - V_old),
            ('beta', np.array(state.beta - beta_old)),
        ])
