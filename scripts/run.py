""" All functions needed to run the model """
import json
import os
import matplotlib.pyplot as plt

from model_solver import *
from model_solver.visualization2 import ConvergenceVisualizer, EquilibriumVisualizer

def make_initial_state(S, I_zones, params, n_edges):
    state0 = ModelState(
    p   =np.ones((S, I_zones)),
    w   =np.ones((S, I_zones)),
    r   =np.ones(I_zones),
    beta=1.0,
    gamma=np.zeros(n_edges),
    Y     =np.zeros((S, I_zones)),
    H_prod=np.zeros((S, I_zones)),
    L_prod=np.zeros((S, I_zones)),
    Q_inter=np.zeros((S, S, I_zones)),
    c     =np.zeros((S, S, I_zones, I_zones)),
    f     =np.zeros((S, I_zones, I_zones)),
    l_res =np.zeros((S, I_zones, I_zones)),

    Q_ship =np.zeros((S, I_zones, I_zones)),
    Q_tilde=np.zeros((S, I_zones, I_zones)),
    tau_eff=np.ones((S, I_zones, I_zones)),
    active =np.zeros((S, I_zones, I_zones), dtype=bool),

    q_pop  =np.full((S, I_zones, I_zones), params.q_bar / (S * I_zones * I_zones)),
    q_bar_s=np.full(S, params.q_bar / S),

    Q_hat              =np.full(n_edges, 0.1),
    q_hat_sj            =np.zeros((S, I_zones, n_edges)),
    Q_hat_sj_goods      =np.zeros((S, I_zones, n_edges)),
    Q_hat_sj_price_goods=np.zeros((S, I_zones, n_edges)),

    I_infra=np.full(n_edges, 1.0),
    gamma_I=np.zeros(n_edges),
    y_k_P=np.zeros(n_edges),
    Q_m_B=np.zeros((1, n_edges)),
    gamma_park=np.zeros(n_edges),
    gamma_board=np.zeros((1, n_edges)),
    V_m=np.zeros((1, n_edges)),
    w_m=np.zeros(1),
    lambda_fleet_m=np.zeros(1),
    lambda_v_cons_mn=np.zeros((1, n_edges)),
    lambda_stat_mn=np.zeros((1, n_edges)),
    h_ijsb=np.zeros((S, I_zones, I_zones, 1)),
    l_bar_b=np.zeros((I_zones, 1)),
    l_tilde_b=np.zeros((I_zones, 1)),
    q0_js=np.zeros((S, I_zones)),
    h0_js=np.zeros((S, I_zones)),
    Q_i0=np.zeros((S, I_zones)),
    Q_0i=np.zeros((S, I_zones)),
    l_tilde_infra=np.zeros(I_zones),
    rho_js=np.zeros((S, I_zones)),
    r_tilde_i=np.zeros(I_zones),
    r_bar_b=np.zeros((I_zones, 1)),
    r_b_r=np.zeros((I_zones, 1)),
    r_j_b_nr=np.zeros((I_zones, 1)),
    r_j_p=np.zeros(I_zones),
    r_i_infra=np.zeros(I_zones),
    pi_j_s=np.zeros((S, I_zones)),
    pi_j_p=np.zeros(I_zones),
    pi_nn_m=np.zeros((1, n_edges)),
    w_j_s=np.zeros((S, I_zones)),
    p_i_s_T=np.zeros((S, I_zones)),
    p_i_s_N=np.zeros((S, I_zones)),
    q_ijs=np.zeros((S, I_zones, I_zones)),
    q_ijsb=np.zeros((S, I_zones, I_zones, 1)),
    q_ijsb_s_k=np.zeros((S, I_zones, I_zones, S, 1)),
    c_ijsb_s=np.zeros((S, S, I_zones, I_zones, 1)),
    d_ijsb_s_k=np.zeros((S, S, I_zones, I_zones, 1)),
    l_ijsb=np.zeros((S, I_zones, I_zones, 1)),
    f_h_ijsb=np.zeros((S, I_zones, I_zones, 1)),
    f_ijsb_s_k=np.zeros((S, I_zones, I_zones, S, 1)),
    H_prod_sb=np.zeros((S, I_zones, 1)),
    L_prod_sb=np.zeros((S, I_zones, 1)),
    H_j_p=np.zeros(I_zones),
    H_nn_m=np.zeros((1, n_edges)),
    l_tilde_m_n=np.zeros((1, n_edges)),
    Y_j_p=np.zeros(I_zones),
    xi_ijsb_k=np.zeros((S, I_zones, I_zones, 1, S)),
    chi_m_nn_ijsb=np.zeros((1, n_edges, S, I_zones, I_zones, 1)),
    mu_ijsb_n=np.zeros((S, I_zones, I_zones, 1, n_edges)),
    mu_ijsb_s_k_n=np.zeros((S, I_zones, I_zones, S, 1, n_edges)),
    nu_s_ij_n=np.zeros((S, I_zones, I_zones, n_edges)),
    mu_0js_n=np.zeros((S, I_zones, n_edges)),
    )
    return state0


def make_network(n_locations, n_modes, type_of_modes, distance_matrix, current_infra_level, capacity, constraints):
        real_world_network = RealWorldNetwork(n_locations = n_locations,
        n_modes = n_modes,
        type_of_modes = type_of_modes,  
        distance_matrix =  distance_matrix,  
        current_infra_level= current_infra_level,   
        capacity = capacity,  
        constraints = constraints)
        return NetworkEncoder(real_world_network).encode()

def visualize_and_save(example_name, final, params, ed, solver):
    """Save solver output and create standard visualizations."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', example_name)
    os.makedirs(output_dir, exist_ok=True)

    final_state, global_conv = final
    # Save numerical state data
    state_path = os.path.join(output_dir, 'final_state.npz')
    np.savez_compressed(
        state_path,
        p=final_state.p,
        w=final_state.w,
        r=final_state.r,
        beta=np.array([final_state.beta]),
        gamma=final_state.gamma,
        I_infra=final_state.I_infra,
        q_pop=final_state.q_pop,
        q_bar_s=final_state.q_bar_s,
        Q_hat=final_state.Q_hat,
    )

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        else:
            return obj

    summary = {
        'example_name': example_name,
        'final_metrics': {
            'beta': float(final_state.beta),
            'max_excess_goods': float(np.abs(ed.ED_goods).max()),
            'max_excess_time': float(np.abs(ed.ED_time).max()),
            'max_excess_land': float(np.abs(ed.ED_land).max()),
            'excess_infra': float(ed.ED_infra),
        },
        'global_conv': global_conv,
        'prices': final_state.p.tolist(),
        'wages': final_state.w.tolist(),
        'rents': final_state.r.tolist(),
        'infrastructure': final_state.I_infra.tolist(),
    }

    summary = to_serializable(summary)
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Plot and save figures
    print(f"\n📊 Saving visualizations into {output_dir}")
    conv_viz = ConvergenceVisualizer(solver.logger)
    fig_conv, _ = conv_viz.plot_convergence()
    fig_conv.savefig(os.path.join(output_dir, 'convergence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_conv)

    eq_viz = EquilibriumVisualizer(final_state, params, solver)
    fig_dash = eq_viz.dashboard()
    fig_dash.savefig(os.path.join(output_dir, 'equilibrium_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_dash)

    fig_prices, _ = eq_viz.plot_prices()
    fig_prices.savefig(os.path.join(output_dir, 'prices.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_prices)

    fig_trade, _ = eq_viz.plot_trade_flows()
    fig_trade.savefig(os.path.join(output_dir, 'trade_flows.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_trade)

    fig_infra, _ = eq_viz.plot_infrastructure()
    fig_infra.savefig(os.path.join(output_dir, 'infrastructure.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_infra)

    fig_pop, _ = eq_viz.plot_population_distribution()
    fig_pop.savefig(os.path.join(output_dir, 'population.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_pop)

    fig_commuters, _ = eq_viz.plot_commuters_network()
    fig_commuters.savefig(os.path.join(output_dir, 'commuters_network.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_commuters)

    flow_viz = FlowVisualizer(final_state, params)

    figf1, _ = flow_viz.plot_commuter_matrices()
    figf1.savefig(os.path.join(output_dir, 'commuters_matrices.png'), dpi=150, bbox_inches='tight')

    figf2, _ = flow_viz.plot_edge_commuter(params.network)
    figf2.savefig(os.path.join(output_dir, 'commuters_edge.png'), dpi=150, bbox_inches='tight')


    figf3, _ = flow_viz.plot_edge_goods(params.network)
    figf3.savefig(os.path.join(output_dir, 'goods_edge.png'), dpi=150, bbox_inches='tight')

    figf4, _ = flow_viz.plot_goods_matrices()
    figf4.savefig(os.path.join(output_dir, 'goods_matrices.png'), dpi=150, bbox_inches='tight')





    print('   ✓ Saved: final_state.npz')
    print('   ✓ Saved: summary.json')
    print('   ✓ Saved: convergence.png')
    print('   ✓ Saved: equilibrium_dashboard.png')
    print('   ✓ Saved: prices.png')
    print('   ✓ Saved: trade_flows.png')
    print('   ✓ Saved: infrastructure.png')
    print('   ✓ Saved: population.png')
    print('   ✓ Saved: commuters_network.png')


def warm_start():
    pass 


def run(
        n_locations, n_modes, types_of_modes, distance_matrix, current_infra_level, capacity, constraints,
        n_sectors, production_function, utility_function, 
        L_bar, H_bar, q_bar, K, 
        sector_dependance_tau, tau_link, t_link, 
        d_w, phi, sigma, sigma_tilde,
        T_outer, eta_p, eta_w, eta_r, T_allocation, eta_beta, eta_I, eta_H, T_inner, alpha_Q, alpha_Qhat, eta_gamma, tol_outer, tol_inner, verbose, log_every, decay_start, decay_type,
        example_name: str, congestion_a: float = 0.6, congestion_b: float = 6.0
    ):

    network = make_network(n_locations, n_modes, types_of_modes, distance_matrix, current_infra_level, capacity, constraints)

    params = ModelParams(n_locations, n_sectors, production_function, utility_function, L_bar, H_bar, q_bar, K, sector_dependance_tau, tau_link(sector_dependance_tau, network.link_dependance_tau, congestion_a, congestion_b), t_link(network.link_dependance_t, congestion_a, congestion_b), d_w, phi, sigma, sigma_tilde, network)
    hyperparams = AlgoParams(T_outer, eta_beta, eta_I, T_allocation, eta_p, eta_w, eta_r, eta_H, T_inner, alpha_Q, alpha_Qhat, eta_gamma, tol_outer, tol_inner, verbose, log_every, decay_start, decay_type)


    state0 = make_initial_state(n_sectors, n_locations, params, params.network.n_edges)
        # Solver
    ed = ExcessDemand()
    pu = PriceUpdater(ed)
    solver = UrbanModelSolver(
        HouseholdBlock(), ProductionBlock(), TransportBlock(),
        InfrastructureBlock(), ed, pu,
    )

    # Solve
    final_state = solver.solve(state0, params, hyperparams)

    print("\n" + "="*60)
    print("SOLVER COMPLETE")
    print("="*60)

    # To complete: add a line of code to save the results in a CSV or json file or the most adequate data structure

    visualize_and_save(example_name, final_state, params, ed, solver)
    return final_state
