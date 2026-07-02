"""The inclusive-value cascade V (Section 4, Steps 0-7).

Step 1 (network potentials) is produced by routing.py and consumed here
as `state.mu_ijsb_n` / `state.mu_ijsb_sk_n`, already evaluated at
iteration t+1 (Remark in Section 4: "these are the freshest available
potentials"). This module implements Steps 2-7: values bottom-up
(eq. 14-18), then quantities top-down (eq. 19-22), per Remark 4 (every
value above a level must be available before any quantity at that level
is computed).

Per the index-set convention fixed in params.py (`n_activities ==
n_sectors`), the activity-purpose price p_k^{s',T}/p_k^{s',N} and output
Y_i^{s'} reuse the production-sector arrays `state.p_sT`/`p_sN`/`Y_js`
directly -- s' and the production-sector index s share one array.
"""
import numpy as np
from scipy.special import logsumexp

from .params import ModelParams
from .state import ModelState

_CLIP = 60.0


class InclusiveValueBlock:
    def solve(self, state: ModelState, params: ModelParams):
        S, I, Br, A = params.n_sectors, params.n_zones, params.n_b_r, params.n_activities
        net = params.network
        trad = params.is_tradable  # (A,) == (S,)

        # ---- Step 2: activity-destination value, eq. 14 ----
        # V_sk[s,i,j,b,k,sp]
        d = state.d_ijsb_sk   # HG + (I, A)
        f = state.f_ijsb_sk   # HG + (I, A)
        p_T_k = state.p_sT.T  # (I, A) -- p_k^{s',T}, k first then s'
        p_N_k = state.p_sN.T  # (I, A)

        # rho_js is indexed by (s, j); j is axis=2 of the household group
        # (s,i,j,b,k,sp), so it is aligned there rather than broadcast as a
        # leading axis.
        rho_b = state.rho_js[:, None, :, None, None, None]  # (S,1,I,1,1,1) aligned to (s,_,j,_,_,_)

        # Evaluated separately on the tradable / non-tradable slices: the
        # concrete primitives' alpha_d/alpha_f arrays are sized to the
        # tradable (resp. non-tradable) activity subset, so they can only
        # broadcast against a same-sized trailing axis.
        flow_term = np.zeros((S, I, I, Br, I, A))
        d_trad, f_trad = d[..., trad], f[..., trad]
        flow_term[..., trad] = (
            params.activity_utility_T.value(d_trad, f_trad)
            - p_T_k[None, None, None, None, :, trad] * d_trad
            - rho_b * f_trad
        )
        f_nontrad = f[..., ~trad]
        flow_term[..., ~trad] = (
            params.activity_utility_N.value(f_nontrad)
            - p_N_k[None, None, None, None, :, ~trad] * f_nontrad
            - rho_b * f_nontrad
        )

        mu_at_k = np.zeros((S, I, I, Br, I, A))
        mu_at_i = np.zeros((S, I, I, Br, I, A))
        for s in range(S):
            for i in range(I):
                i_node = net.zone_to_node(i)
                for j in range(I):
                    for b in range(Br):
                        for k in range(I):
                            k_node = net.zone_to_node(k)
                            mu_at_k[s, i, j, b, k, :] = state.mu_ijsb_sk_n[s, i, j, b, k, :, k_node]
                            mu_at_i[s, i, j, b, k, :] = state.mu_ijsb_sk_n[s, i, j, b, k, :, i_node]

        # routing.py's mu is stored "cost-to-destination" (mu_destination=0,
        # mu_origin=full trip cost), the mirror of eq. 5/7/8's implicit
        # "cost-from-origin" convention (mu_origin=0, mu_destination=full
        # cost) -- see the note in inner_loop.py's eq. 37 for the derivation.
        # Every document term (mu_X - mu_Y) becomes (mu_Y - mu_X) here.
        V_sk = flow_term + (mu_at_k - mu_at_i)
        self.V_sk = V_sk

        # ---- Step 3: activity-sector inclusive value, eq. 15 ----
        sigma_sp = params.sigma_sprime  # (A,)
        scaled = np.clip(V_sk / sigma_sp[None, None, None, None, None, :], -_CLIP, _CLIP)
        A_sp = sigma_sp[None, None, None, None, :] * logsumexp(scaled, axis=4)  # sum over k -> (S,I,I,Br,A)
        self.A_sprime = A_sp

        # ---- Step 4: building/residence-level value, eq. 16 ----
        c = state.c_ijsb_s  # HGA = (S,I,I,Br,A)
        c_trad = c[..., trad]                       # (S,I,I,Br,n_trad)
        p_T_home = state.p_sT.T[..., trad]           # (I, n_trad) -- p_i^{s',T}, home zone i
        U_val = params.utility.value(c_trad, state.l_ijsb, state.fh_ijsb)  # (S,I,I,Br)
        c_cost = np.einsum('sijba,ia->sijb', c_trad, p_T_home)

        mu_n_at_j = np.zeros((S, I, I, Br))
        mu_n_at_i = np.zeros((S, I, I, Br))
        for s in range(S):
            for i in range(I):
                i_node = net.zone_to_node(i)
                for j in range(I):
                    j_node = net.zone_to_node(j)
                    mu_n_at_j[s, i, j, :] = state.mu_ijsb_n[s, i, j, :, j_node]
                    mu_n_at_i[s, i, j, :] = state.mu_ijsb_n[s, i, j, :, i_node]

        r_res = state.r_res[None, :, None, :]          # (1,I,1,Br) aligned to (_,i,_,b)
        rho_jb = state.rho_js[:, None, :, None]         # (S,1,I,1) aligned to (s,_,j,_)
        w_jb = state.w_js[:, None, :, None]              # (S,1,I,1)
        omega_A_sum = np.einsum('a,sijba->sijb', params.omega_sprime, A_sp)

        # (mu_n_at_j - mu_n_at_i) sign: see the note above V_sk / in
        # inner_loop.py's eq. 37 -- routing.py's stored mu is the mirror of
        # the document's implicit convention, so the document's -(mu_j-mu_i)
        # becomes +(mu_j-mu_i) here.
        V_ijsb = (U_val - r_res * state.l_ijsb
                  - c_cost
                  + (mu_n_at_j - mu_n_at_i)
                  - rho_jb * (state.h_ijsb + state.fh_ijsb - params.H_bar)
                  + w_jb * state.h_ijsb
                  + omega_A_sum)
        self.V_ijsb = V_ijsb

        # ---- Step 5: residence-workplace inclusive value, eq. 17 ----
        sigma_loc = params.sigma_loc[:, None, None]  # (S,1,1)
        scaled_b = np.clip(V_ijsb / sigma_loc[..., None], -_CLIP, _CLIP)
        V_ijs = params.sigma_loc[:, None, None] * logsumexp(scaled_b, axis=3)  # sum over b -> (S,I,I)
        self.V_ijs = V_ijs

        # ---- Step 6: sector inclusive value, eq. 18 ----
        sigma_s = params.sigma_s[:, None, None]
        scaled_s = np.clip(V_ijs / sigma_s, -_CLIP, _CLIP)
        V_s = params.sigma_s * logsumexp(scaled_s.reshape(S, -1), axis=1)  # sum over (i,j) -> (S,)
        self.V_s = V_s

        # ---- Step 7: quantities, top-down, eq. 19-22 ----
        scaled_tilde = np.clip(V_s / params.sigma_tilde, -_CLIP, _CLIP)
        log_norm = logsumexp(scaled_tilde)
        q_bar_s = params.q_bar * np.exp(np.clip(scaled_tilde - log_norm, -_CLIP, _CLIP))
        state.q_bar_s = q_bar_s

        log_norm_ij = logsumexp(scaled_s.reshape(S, -1), axis=1)  # (S,)
        q_ijs = q_bar_s[:, None, None] * np.exp(
            np.clip(scaled_s - log_norm_ij[:, None, None], -_CLIP, _CLIP))
        state.q_ijs = q_ijs

        log_norm_b = logsumexp(scaled_b, axis=3)  # (S,I,I)
        q_ijsb = q_ijs[:, :, :, None] * np.exp(
            np.clip(scaled_b - log_norm_b[:, :, :, None], -_CLIP, _CLIP))
        state.q_ijsb = q_ijsb

        log_norm_sk = logsumexp(scaled, axis=4)  # (S,I,I,Br,A)
        q_ijsb_sk = (params.omega_sprime[None, None, None, None, None, :]
                     * q_ijsb[:, :, :, :, None, None]
                     * np.exp(np.clip(scaled - log_norm_sk[:, :, :, :, None, :], -_CLIP, _CLIP)))
        state.q_ijsb_sk = q_ijsb_sk
