import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, List

from .models import Network

try:
    import networkx as nx
except ImportError:
    nx = None


def _apply_clean_style():
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'grid.alpha': 0.25,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
    })


class NetworkDecoder:
    """Decode edge metadata from a multimodal Network object."""

    def __init__(self, network: Network, mode_names: Optional[List[str]] = None):
        self.network = network
        self.mode_names = mode_names or [f'M{m}' for m in range(self.network.n_modes)]
        self.original_edges = list(network.original_edges or [])
        self.n_original = len(self.original_edges)
        self.edge_metadata = self._build_edge_metadata()

    def _build_edge_metadata(self) -> List[Dict]:
        metadata: List[Dict] = []
        if self.n_original == 0:
            return metadata

        n_modes = self.network.n_modes
        n_intra_edges = self.n_original * n_modes

        for edge_idx in range(n_intra_edges):
            mode_idx = edge_idx // self.n_original
            pair_idx = edge_idx % self.n_original
            origin, destination = self.original_edges[pair_idx]
            metadata.append({
                'edge_index': edge_idx,
                'origin_zone': origin,
                'destination_zone': destination,
                'origin_node': origin + mode_idx * self.network.n_zones,
                'destination_node': destination + mode_idx * self.network.n_zones,
                'mode': self.mode_names[mode_idx],
                'is_transfer': False,
                'transfer_from': None,
                'transfer_to': None,
            })

        if n_modes > 1:
            n_transfers = self.network.n_zones * n_modes * (n_modes - 1)
            for transfer_idx in range(n_transfers):
                edge_idx = n_intra_edges + transfer_idx
                zone = transfer_idx // (n_modes * (n_modes - 1))
                offset = transfer_idx % (n_modes * (n_modes - 1))
                from_mode = offset // (n_modes - 1)
                to_mode = offset % (n_modes - 1)
                if to_mode >= from_mode:
                    to_mode += 1
                origin_node = zone * n_modes + from_mode
                destination_node = zone * n_modes + to_mode
                metadata.append({
                    'edge_index': edge_idx,
                    'origin_zone': zone,
                    'destination_zone': zone,
                    'origin_node': origin_node,
                    'destination_node': destination_node,
                    'mode': f'{self.mode_names[from_mode]}→{self.mode_names[to_mode]}',
                    'is_transfer': True,
                    'transfer_from': self.mode_names[from_mode],
                    'transfer_to': self.mode_names[to_mode],
                })
        return metadata

    def decode_edge(self, edge_index: int) -> Dict:
        return self.edge_metadata[edge_index]

    def edge_labels(self, short: bool = True) -> List[str]:
        labels = []
        for meta in self.edge_metadata:
            if meta['is_transfer']:
                label = f'{meta["origin_zone"]}:{meta["mode"]}'
            else:
                label = f'{meta["origin_zone"]}->{meta["destination_zone"]}:{meta["mode"]}'
            labels.append(label if not short else label.replace(' ', ''))
        return labels

    def aggregate_by_zone_pair(self, values: np.ndarray, include_transfers: bool = False) -> np.ndarray:
        matrix = np.zeros((self.network.n_zones, self.network.n_zones), dtype=float)
        values = np.asarray(values, dtype=float)
        for idx, value in enumerate(values):
            meta = self.edge_metadata[idx]
            if meta['is_transfer'] and not include_transfers:
                continue
            matrix[meta['origin_zone'], meta['destination_zone']] += value
        return matrix


class ConvergenceVisualizer:
    """Clean convergence plots for outer and inner solver metrics."""

    def __init__(self, outer_logger=None, inner_logger=None):
        self.outer_logger = outer_logger
        self.inner_logger = inner_logger
        _apply_clean_style()

    def plot_convergence(self, figsize=(14, 14)):
        if self.outer_logger is None or not hasattr(self.outer_logger, 'history'):
            raise ValueError('Outer logger with history is required for convergence plots.')

        fig, axes = plt.subplots(3, 2, figsize=figsize)
        history = self.outer_logger.history

        # Residuals
        axes[0, 0].semilogy(history.get('residuals', []), linewidth=2, color='#2a9d8f')
        axes[0, 0].set_title('Outer Loop: Combined Excess Demand')
        axes[0, 0].set_xlabel('Outer iteration')
        axes[0, 0].set_ylabel('Norm of residuals')
        axes[0, 0].grid(True, alpha=0.3)
       # Price changes
        axes[0, 1].semilogy(history.get('price_changes', []), linewidth=2, color='#e76f51')
        axes[0, 1].set_title('Outer Loop: Total Price Change')
        axes[0, 1].set_xlabel('Outer iteration')
        axes[0, 1].set_ylabel('Price change')
        axes[0, 1].grid(True, alpha=0.3) 

        # Prices evolution: all price components over outer iterations
        prices_history = history.get('prices', [])
        price_lines = []
        price_labels = []

        for t, price_tuple in enumerate(prices_history):
            if not isinstance(price_tuple, (list, tuple)) or len(price_tuple) != 3:
                raise ValueError('Expected prices to be stored as [p, w, r] for each outer iteration.')
            p, w, r = price_tuple
            p = np.asarray(p, dtype=float)
            w = np.asarray(w, dtype=float)
            r = np.asarray(r, dtype=float)

            if t == 0:
                for si in np.ndindex(p.shape):
                    price_labels.append(f'p{si}')
                for si in np.ndindex(w.shape):
                    price_labels.append(f'w{si}')
                for si in np.ndindex(r.shape):
                    price_labels.append(f'r{si}')
                price_lines = [[] for _ in price_labels]

            flat_prices = np.concatenate([p.ravel(), w.ravel(), r.ravel()])
            if t == 0:
                expected_size = flat_prices.size
            elif flat_prices.size != expected_size:
                raise ValueError('Inconsistent price dimensions across outer iterations.')

            for idx, val in enumerate(flat_prices):
                price_lines[idx].append(val)

        for idx, series in enumerate(price_lines):
            axes[1, 0].plot(series, linewidth=1.5, alpha=0.85, label=price_labels[idx])

        axes[1, 0].set_title('Evolution of All Prices')
        axes[1, 0].set_xlabel('Outer iteration')
        axes[1, 0].set_ylabel('Price value')
        if len(price_lines) <= 12:
            axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Production and Aggregate Flows evolution (norms)
        production_norms = [np.linalg.norm(prod) for prod in history.get('production', [])]
        flow_history = [np.asarray(flows, dtype=float) for flows in history.get('aggregate_flows', [])]
        flows_norms = [np.linalg.norm(flows) for flows in flow_history]
        axes[1, 1].plot(production_norms, linewidth=2, label='Production', color='#f4a261')
        axes[1, 1].plot(flows_norms, linewidth=2, label='Aggregate Flows', color='#e9c46a')
        axes[1, 1].set_title('Evolution of Production and Flows (Norms)')
        axes[1, 1].set_xlabel('Outer iteration')
        axes[1, 1].set_ylabel('Norm')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Aggregate flow diagnostics: edge evolution and change magnitude
        if flow_history:
            edge_count = flow_history[0].size
            if edge_count <= 12:
                for edge_idx in range(edge_count):
                    edge_series = [flows[edge_idx] for flows in flow_history]
                    axes[2, 0].plot(edge_series, linewidth=1.3, alpha=0.8, label=f'edge {edge_idx}')
                axes[2, 0].legend(loc='upper right', fontsize=8)
            else:
                axes[2, 0].plot(flows_norms, linewidth=2, color='#2a9d8f')
                axes[2, 0].set_ylabel('Norm')
            axes[2, 0].set_title('Aggregate Flow Evolution per Edge')
            axes[2, 0].set_xlabel('Outer iteration')
            axes[2, 0].grid(True, alpha=0.3)

            flow_changes = [np.linalg.norm(flow_history[i] - flow_history[i - 1]) for i in range(1, len(flow_history))]
            axes[2, 1].plot(flow_changes, linewidth=2, color='#8a3ffc')
            axes[2, 1].set_title('Aggregate Flow Changes Between Iterations')
            axes[2, 1].set_xlabel('Outer iteration')
            axes[2, 1].set_ylabel('Change norm')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'No flow history available', ha='center', va='center')
            axes[2, 0].axis('off')
            axes[2, 1].text(0.5, 0.5, 'No flow history available', ha='center', va='center')
            axes[2, 1].axis('off')

        plt.tight_layout()
        return fig, axes

    def plot_inner_convergence(self, figsize=(12, 4)):
        if self.inner_logger is None or not hasattr(self.inner_logger, 'history'):
            raise ValueError('Inner logger with history is required for inner convergence plots.')

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        history = self.inner_logger.history
        ax.semilogy(history.get('flow_changes', []), linewidth=2, label='ΔQ')
        ax.semilogy(history.get('congestion_changes', []), linewidth=2, label='Δγ')
        ax.set_title('Inner Loop Convergence')
        ax.set_xlabel('Inner iteration')
        ax.set_ylabel('Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax


class EquilibriumVisualizer:
    """Clean dashboard and focused plots for equilibrium results."""

    def __init__(
        self,
        final_state,
        params,
        solver=None,
        zone_pos: Optional[Dict] = None,
        zone_names: Optional[List[str]] = None,
        sector_names: Optional[List[str]] = None,
        mode_names: Optional[List[str]] = None,
    ):
        self.state = final_state
        self.params = params
        self.solver = solver
        self.S = params.n_sectors
        self.I = params.n_zones
        self.net = params.network
        self.decoder = NetworkDecoder(self.net, mode_names=mode_names)
        self.zone_names = zone_names or [f'Z{i}' for i in range(self.I)]
        self.sector_names = sector_names or [f'S{i}' for i in range(self.S)]
        self.mode_names = mode_names or [f'M{i}' for i in range(self.net.n_modes)]
        self.zone_pos = zone_pos or self._auto_layout()
        _apply_clean_style()

    def _auto_layout(self) -> Dict[int, tuple]:
        side = int(np.round(np.sqrt(self.I)))
        if side * side == self.I:
            return {i: (i % side, -(i // side)) for i in range(self.I)}
        return {i: (i, 0) for i in range(self.I)}

    def dashboard(self, figsize=(16, 12)):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

        ax_prices = fig.add_subplot(gs[0, 0])
        ax_wages = fig.add_subplot(gs[0, 1])
        ax_rents = fig.add_subplot(gs[1, 0])
        ax_infra = fig.add_subplot(gs[1, 1])
        ax_congestion = fig.add_subplot(gs[2, 0])
        ax_population = fig.add_subplot(gs[2, 1])

        self._plot_price_matrix(ax_prices)
        self._plot_wage_map(ax_wages)
        self._plot_rents(ax_rents)
        self._plot_aggregated_infrastructure(ax_infra)
        self._plot_aggregated_congestion(ax_congestion)
        self._plot_population(ax_population)

        fig.suptitle('Equilibrium Dashboard', fontsize=18, y=0.98, fontweight='bold')
        return fig

    def _plot_price_matrix(self, ax):
        im = ax.imshow(self.state.p, cmap='RdYlGn_r', aspect='auto')
        ax.set_title('Good Prices p[s, i]')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Sector')
        ax.set_xticks(range(self.I))
        ax.set_xticklabels(self.zone_names)
        ax.set_yticks(range(self.S))
        ax.set_yticklabels(self.sector_names)
        plt.colorbar(im, ax=ax, fraction=0.05)

    def _plot_wage_map(self, ax):
        im = ax.imshow(self.state.w, cmap='viridis', aspect='auto')
        ax.set_title('Wages w[s, i]')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Sector')
        ax.set_xticks(range(self.I))
        ax.set_xticklabels(self.zone_names)
        ax.set_yticks(range(self.S))
        ax.set_yticklabels(self.sector_names)
        plt.colorbar(im, ax=ax, fraction=0.05)

    def _plot_rents(self, ax):
        rent_values = self.state.r
        ax.bar(range(self.I), rent_values, color='#264653', alpha=0.85)
        ax.set_title('Rents by Zone')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Rent')
        ax.set_xticks(range(self.I))
        ax.set_xticklabels(self.zone_names)
        ax.grid(True, alpha=0.25)

    def _plot_aggregated_infrastructure(self, ax):
        infra_by_pair = self.decoder.aggregate_by_zone_pair(self.state.I_infra)
        im = ax.imshow(infra_by_pair, cmap='Blues', aspect='auto')
        ax.set_title('Aggregated Infrastructure by Zone Pair')
        ax.set_xlabel('Destination')
        ax.set_ylabel('Origin')
        ax.set_xticks(range(self.I))
        ax.set_xticklabels(self.zone_names)
        ax.set_yticks(range(self.I))
        ax.set_yticklabels(self.zone_names)
        plt.colorbar(im, ax=ax, fraction=0.05)

    def _plot_aggregated_congestion(self, ax):
        gamma_by_pair = self.decoder.aggregate_by_zone_pair(self.state.gamma)
        im = ax.imshow(gamma_by_pair, cmap='OrRd', aspect='auto')
        ax.set_title('Aggregated Congestion by Zone Pair')
        ax.set_xlabel('Destination')
        ax.set_ylabel('Origin')
        ax.set_xticks(range(self.I))
        ax.set_xticklabels(self.zone_names)
        ax.set_yticks(range(self.I))
        ax.set_yticklabels(self.zone_names)
        plt.colorbar(im, ax=ax, fraction=0.05)

    def _plot_population(self, ax):
        origin_population = self.state.q_pop.sum(axis=(0, 2))
        destination_population = self.state.q_pop.sum(axis=(0, 1))
        x = np.arange(self.I)
        ax.plot(x, origin_population, marker='o', label='Live in zone', color='#2a9d8f')
        ax.plot(x, destination_population, marker='s', label='Work in zone', color='#e76f51')
        ax.set_title('Population by Zone')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Population')
        ax.set_xticks(x)
        ax.set_xticklabels(self.zone_names)
        ax.legend()
        ax.grid(True, alpha=0.25)

    def plot_prices(self, figsize=(14, 5)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_price_matrix(axes[0])
        self._plot_wage_map(axes[1])
        plt.tight_layout()
        return fig, axes

    def plot_trade_flows(self, figsize=(12, 5)):
        fig, axes = plt.subplots(1, self.S, figsize=figsize)
        if self.S == 1:
            axes = [axes]
        for s in range(self.S):
            trade_matrix = self.state.Q_ship[s]
            im = axes[s].imshow(trade_matrix, cmap='Blues', aspect='auto')
            axes[s].set_title(f'Trade Matrix: {self.sector_names[s]}')
            axes[s].set_xlabel('Destination Zone')
            axes[s].set_ylabel('Origin Zone')
            axes[s].set_xticks(range(self.I))
            axes[s].set_xticklabels(self.zone_names)
            axes[s].set_yticks(range(self.I))
            axes[s].set_yticklabels(self.zone_names)
            plt.colorbar(im, ax=axes[s], fraction=0.05)
        plt.tight_layout()
        return fig, axes

    def plot_infrastructure(self, figsize=(14, 5)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_infrastructure_bars(axes[0])
        self._plot_aggregated_infrastructure(axes[1])
        plt.tight_layout()
        return fig, axes

    def _plot_infrastructure_bars(self, ax):
        labels = self.decoder.edge_labels(short=True)
        values = self.state.I_infra
        colors = ['#2a9d8f' if not meta['is_transfer'] else '#8d99ae' for meta in self.decoder.edge_metadata]
        x = np.arange(len(values))
        ax.bar(x, values, color=colors, alpha=0.85)
        ax.set_title('Infrastructure Levels by Edge')
        ax.set_xlabel('Edge index')
        ax.set_ylabel('I_infra')
        ax.grid(True, alpha=0.2)
        if len(values) <= 25:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            step = max(1, len(values) // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([labels[i] for i in x[::step]], rotation=45, ha='right')

    def plot_population_distribution(self, figsize=(12, 5)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        origin_population = self.state.q_pop.sum(axis=(0, 2))
        axes[0].bar(range(self.I), origin_population, color='#2a9d8f', alpha=0.9)
        axes[0].set_title('Population by Home Zone')
        axes[0].set_xlabel('Zone')
        axes[0].set_ylabel('Population')
        axes[0].set_xticks(range(self.I))
        axes[0].set_xticklabels(self.zone_names)
        axes[0].grid(True, alpha=0.25)

        sector_population = self.state.q_bar_s
        axes[1].bar(range(self.S), sector_population, color='#e76f51', alpha=0.9)
        axes[1].set_title('Population by Sector')
        axes[1].set_xlabel('Sector')
        axes[1].set_ylabel('Population')
        axes[1].set_xticks(range(self.S))
        axes[1].set_xticklabels(self.sector_names)
        axes[1].grid(True, alpha=0.25)

        plt.tight_layout()
        return fig, axes

    def plot_commuters_network(self, figsize=(10, 10), min_flow_threshold=0.0):
        if nx is None:
            raise ImportError('networkx is required for network visualizations.')

        total_commuters = self.state.q_pop.sum(axis=0)
        G = nx.DiGraph()
        for i in range(self.I):
            G.add_node(i)
        for i in range(self.I):
            for j in range(self.I):
                if i == j or total_commuters[i, j] <= min_flow_threshold:
                    continue
                G.add_edge(i, j, weight=total_commuters[i, j])

        fig, ax = plt.subplots(figsize=figsize)
        positions = self.zone_pos
        node_labels = {i: self.zone_names[i] for i in range(self.I)}
        nx.draw_networkx_nodes(G, positions, node_color='#2a9d8f', node_size=700, ax=ax)
        nx.draw_networkx_labels(G, positions, labels=node_labels, ax=ax, font_size=10)
        if G.number_of_edges() > 0:
            widths = np.array([G[u][v]['weight'] for u, v in G.edges()])
            widths = 0.5 + 4.5 * (widths - widths.min()) / (widths.max() - widths.min() + 1e-10)
            nx.draw_networkx_edges(
                G,
                positions,
                width=widths,
                edge_color='#264653',
                arrowsize=16,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                alpha=0.8,
                ax=ax,
            )
        ax.set_title('Commuter Network')
        ax.set_axis_off()
        plt.tight_layout()
        return fig, ax


class FlowVisualizer:
    """Focused flow plots for commuters and goods."""

    def __init__(self, state, params, zone_names: Optional[List[str]] = None, sector_names: Optional[List[str]] = None):
        self.state = state
        self.params = params
        self.zone_names = zone_names or [f'Z{i}' for i in range(params.n_zones)]
        self.sector_names = sector_names or [f'S{i}' for i in range(params.n_sectors)]
        _apply_clean_style()

    def plot_commuter_matrices(self, figsize=(14, 4)):
        fig, axes = plt.subplots(1, self.params.n_sectors, figsize=figsize)
        if self.params.n_sectors == 1:
            axes = [axes]
        for s in range(self.params.n_sectors):
            im = axes[s].imshow(self.state.q_pop[s], cmap='Purples', aspect='auto')
            axes[s].set_title(f'Commuters: {self.sector_names[s]}')
            axes[s].set_xlabel('Destination Zone')
            axes[s].set_ylabel('Origin Zone')
            axes[s].set_xticks(range(self.params.n_zones))
            axes[s].set_xticklabels(self.zone_names)
            axes[s].set_yticks(range(self.params.n_zones))
            axes[s].set_yticklabels(self.zone_names)
            plt.colorbar(im, ax=axes[s], fraction=0.05)
        plt.tight_layout()
        return fig, axes

    def plot_goods_matrices(self, figsize=(14, 4)):
        fig, axes = plt.subplots(1, self.params.n_sectors, figsize=figsize)
        if self.params.n_sectors == 1:
            axes = [axes]
        for s in range(self.params.n_sectors):
            im = axes[s].imshow(self.state.Q_ship[s], cmap='Blues', aspect='auto')
            axes[s].set_title(f'Goods Shipments: {self.sector_names[s]}')
            axes[s].set_xlabel('Destination Zone')
            axes[s].set_ylabel('Origin Zone')
            axes[s].set_xticks(range(self.params.n_zones))
            axes[s].set_xticklabels(self.zone_names)
            axes[s].set_yticks(range(self.params.n_zones))
            axes[s].set_yticklabels(self.zone_names)
            plt.colorbar(im, ax=axes[s], fraction=0.05)
        plt.tight_layout()
        return fig, axes

    def plot_edge_infrastructure(self, network: Network, figsize=(14, 4)):
        decoder = NetworkDecoder(network)
        labels = decoder.edge_labels(short=True)
        values = self.state.I_infra
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(range(len(values)), values, color='#2a9d8f', alpha=0.85)
        ax.set_title('Infrastructure by Transport Edge')
        ax.set_xlabel('Edge index')
        ax.set_ylabel('I_infra')
        if len(values) <= 25:
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            step = max(1, len(values) // 20)
            ax.set_xticks(range(0, len(values), step))
            ax.set_xticklabels([labels[i] for i in range(0, len(values), step)], rotation=45, ha='right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig, ax
    
    def plot_edge_commuter(self, network: Network, figsize=(14, 4)):
        decoder = NetworkDecoder(network)
        labels = decoder.edge_labels(short=True)
        values = np.sum(self.state.q_hat_sj, axis=(0,1))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(range(len(values)), values, color='#2a9d8f', alpha=0.85)
        ax.set_title('Commuters flow by Transport Edge')
        ax.set_xlabel('Edge index')
        ax.set_ylabel('q_hat')
        if len(values) <= 25:
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            step = max(1, len(values) // 20)
            ax.set_xticks(range(0, len(values), step))
            ax.set_xticklabels([labels[i] for i in range(0, len(values), step)], rotation=45, ha='right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig, ax
    
    def plot_edge_goods(self, network: Network, figsize=(14, 4)):
        decoder = NetworkDecoder(network)
        labels = decoder.edge_labels(short=True)
        values = np.sum(self.state.Q_hat_sj_goods, axis=(0,1))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(range(len(values)), values, color='#2a9d8f', alpha=0.85)
        ax.set_title('Goods flow by Transport Edge')
        ax.set_xlabel('Edge index')
        ax.set_ylabel('Q_hat_goods')
        if len(values) <= 25:
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
        else:
            step = max(1, len(values) // 20)
            ax.set_xticks(range(0, len(values), step))
            ax.set_xticklabels([labels[i] for i in range(0, len(values), step)], rotation=45, ha='right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        return fig, ax


class VisualizeEvolution():

    def __init__(self, state, params):
        self.state = state
        self.params = params
