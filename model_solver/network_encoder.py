from dataclasses import dataclass
import numpy as np
from .models import Network

class ModeOfTransport:
    def __init__(self, name, speed, cost_per_km):
        self.name = name
        self.speed = speed
        self.cost_per_km = cost_per_km

@dataclass
class RealWorldNetwork:
    """ real world network data structure to be encoded for the model """
    n_locations: int  # number of locations that are connected by the transport network
    n_modes: int  # number of transport modes
    type_of_modes: list[ModeOfTransport]  # list of transport modes
    distance_matrix: np.ndarray  # matrix of distances between locations (shape: n_locations x n_locations)
    current_infra_level: np.ndarray  # shape: (n_modes, n_locations , n_locations) 
    capacity: np.ndarray  # shape: (n_modes, n_locations , n_locations) 
    constraints: np.ndarray  # shape: (n_modes, 2): [goods_allowed, commuters_allowed]

    def __post_init__(self):
        self.distance_matrix = np.asarray(self.distance_matrix, dtype=float)
        self.constraints = np.asarray(self.constraints, dtype=bool)
        self.current_infra_level = np.asarray(self.current_infra_level, dtype=float)
        self.capacity = np.asarray(self.capacity, dtype=float)
        if self.current_infra_level.ndim == 1:
            self.current_infra_level = self.current_infra_level.reshape(
            self.n_modes, self.n_locations, self.n_locations
        )
        if self.capacity.ndim == 1:
            self.capacity = self.capacity.reshape(
                self.n_modes, self.n_locations, self.n_locations
            )


class NetworkEncoder:
    def __init__(self, real_world_network: RealWorldNetwork):
        self.network = real_world_network
        self.edges = []
        self.original_edges = []
        self.n_nodes = self.network.n_locations * self.network.n_modes

    def _zone_node(self, zone: int, mode: int) -> int:
        return int(zone + mode * self.network.n_locations)

    def create_edges(self):
        n = self.network.n_locations
        m = self.network.n_modes
        edges = []
        original = []

        for mode in range(m):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    u = self._zone_node(i, mode)
                    v = self._zone_node(j, mode)
                    edges.append((u, v))
                    original.append((i, j))

        if m > 1:
            for zone in range(n):
                for from_mode in range(m):
                    for to_mode in range(m):
                        if from_mode == to_mode:
                            continue
                        u = self._zone_node(zone, from_mode)
                        v = self._zone_node(zone, to_mode)
                        edges.append((u, v))

        self.edges = edges
        self.original_edges = list(dict.fromkeys(original))

    def create_time_paramters(self) -> np.ndarray:
        n = self.network.n_locations
        m = self.network.n_modes
        distances = np.asarray(self.network.distance_matrix, dtype=float)
        T0 = []
        large = 1e20

        for mode in range(m):
            goods_allowed, commuters_allowed = self.network.constraints[mode]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if commuters_allowed:
                        T0.append(distances[i, j] / self.network.type_of_modes[mode].speed)
                    else:
                        T0.append(large)

        if m > 1:
            for zone in range(n):
                for from_mode in range(m):
                    for to_mode in range(m):
                        if from_mode == to_mode:
                            continue
                        _, commuters_allowed_from = self.network.constraints[from_mode]
                        _, commuters_allowed_to = self.network.constraints[to_mode]
                        if commuters_allowed_from and commuters_allowed_to:
                            T0.append(1.0)
                        else:
                            T0.append(large)

        return np.asarray(T0, dtype=float)

    def create_iceberg_cost_parameters(self) -> np.ndarray:
        n = self.network.n_locations
        m = self.network.n_modes
        mask = []
        tau_scale = []
        distances = np.asarray(self.network.distance_matrix, dtype=float)

        for mode in range(m):
            goods_allowed, _ = self.network.constraints[mode]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    mask.append(1.0 if goods_allowed else 0.0)
                    tau_scale.append(distances[i, j] / self.network.type_of_modes[mode].speed)

        if m > 1:
            for zone in range(n):
                for from_mode in range(m):
                    for to_mode in range(m):
                        if from_mode == to_mode:
                            continue
                        goods_allowed_from, _ = self.network.constraints[from_mode]
                        goods_allowed_to, _ = self.network.constraints[to_mode]
                        mask.append(1.0 if goods_allowed_from and goods_allowed_to else 0.0)
                        tau_scale.append(0.0)

        return np.asarray(mask, dtype=float), np.asarray(tau_scale, dtype=float)

    def create_infra_bounds(self) -> np.ndarray:
        n = self.network.n_locations
        m = self.network.n_modes
        I_min = []
        I_max = []

        volumes = np.asarray(self.network.current_infra_level, dtype=float)
        capacities = np.asarray(self.network.capacity, dtype=float)

        for mode in range(m):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    I_min.append(volumes[mode, i, j])
                    I_max.append(capacities[mode, i, j])

        if m > 1:
            for zone in range(n):
                for from_mode in range(m):
                    for to_mode in range(m):
                        if from_mode == to_mode:
                            continue
                        I_min.append(1.0)
                        I_max.append(5.0)

        return np.vstack([np.asarray(I_min, dtype=float), np.asarray(I_max, dtype=float)]).T

    def create_infra_cost_parameters(self) -> np.ndarray:
        n = self.network.n_locations
        m = self.network.n_modes
        kappa = []

        for mode in range(m):
            cost = self.network.type_of_modes[mode].cost_per_km
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    kappa.append(cost)

        if m > 1:
            for zone in range(n):
                for from_mode in range(m):
                    for to_mode in range(m):
                        if from_mode == to_mode:
                            continue
                        kappa.append(0.5 * (
                            self.network.type_of_modes[from_mode].cost_per_km
                            + self.network.type_of_modes[to_mode].cost_per_km
                        ))

        return np.asarray(kappa, dtype=float)

    def encode(self) -> Network:
        self.create_edges()
        edges = self.edges
        T0 = self.create_time_paramters()
        tau_mask, tau_scale = self.create_iceberg_cost_parameters()
        infra_bounds = self.create_infra_bounds()
        kappa = self.create_infra_cost_parameters()

        zone_to_nodes = np.asarray([
            [self._zone_node(zone, mode) for mode in range(self.network.n_modes)]
            for zone in range(self.network.n_locations)
        ], dtype=int)
        mode_to_nodes = np.asarray([
            [self._zone_node(zone, mode) for zone in range(self.network.n_locations)]
            for mode in range(self.network.n_modes)
        ], dtype=int)

        return Network(
            n_zones=self.network.n_locations,
            n_nodes=self.n_nodes,
            edges=self.edges,
            n_modes=self.network.n_modes,
            n_edges=len(edges),
            kappa=kappa,
            I_min=infra_bounds[:, 0].astype(float),
            I_max=infra_bounds[:, 1].astype(float),
            link_dependance_tau=(tau_mask, tau_scale),
            link_dependance_t=T0,
            zone_to_nodes=zone_to_nodes,
            zone_to_node=zone_to_nodes,
            mode_to_nodes=mode_to_nodes,
            original_edges = self.original_edges
        )


