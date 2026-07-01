"""Transport network graph: nodes, edges, zones, transit-mode link
membership, and the single boundary/entry node N^G = {n_g} (Section 3).

Convention: every zone i in I is itself a graph node (node indices
0..n_zones-1); one extra node is appended as the boundary node n_g used
for peripheral commuters / freight import-export. Edges carry an L^car
flag (car-usable links, referenced in eq. 5/8/32) and, for transit
edges, a transit-mode index used to build the boolean L^m membership
mask (eq. 5/6, used wherever a link cost is restricted to mode m).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Network:
    n_zones: int
    n_nodes: int                      # n_zones zone-nodes + 1 boundary node
    n_modes_transit: int
    edges: List[Tuple[int, int]]
    n_edges: int = field(init=False)
    is_car: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))      # (n_edges,)  L^car
    mode_of_edge: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))  # (n_edges,), -1 if not transit
    freight_allowed: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))  # (n_edges,)
    kappa: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))      # (n_edges,) infra unit cost
    I_min: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    I_max: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    entry_node: int = 0

    adj_out: List[List[Tuple[int, int]]] = field(init=False)
    adj_in: List[List[Tuple[int, int]]] = field(init=False)
    L_m: np.ndarray = field(init=False)  # (n_modes_transit, n_edges) boolean

    def __post_init__(self):
        self.n_edges = len(self.edges)
        self.adj_out = [[] for _ in range(self.n_nodes)]
        self.adj_in = [[] for _ in range(self.n_nodes)]
        for e, (u, v) in enumerate(self.edges):
            if not (0 <= u < self.n_nodes and 0 <= v < self.n_nodes):
                raise ValueError(f"Edge {(u, v)} references node outside [0, {self.n_nodes})")
            self.adj_out[u].append((v, e))
            self.adj_in[v].append((u, e))
        self.L_m = np.zeros((self.n_modes_transit, self.n_edges), dtype=bool)
        for m in range(self.n_modes_transit):
            self.L_m[m] = (self.mode_of_edge == m)

    def zone_to_node(self, zone: int) -> int:
        """Zones map 1:1 to graph nodes 0..n_zones-1."""
        return zone


class NetworkBuilder:
    """Small helper to assemble a Network from a flat edge specification,
    mirroring model_solver/network_encoder.py's role (a thin translation
    layer from "real world" link lists into the solver's Network)."""

    def __init__(self, n_zones: int, n_modes_transit: int):
        self.n_zones = n_zones
        self.n_modes_transit = n_modes_transit
        self.entry_node = n_zones  # boundary node n_g, appended after the zone nodes
        self.n_nodes = n_zones + 1
        self._edges: List[Tuple[int, int]] = []
        self._is_car: List[bool] = []
        self._mode_of_edge: List[int] = []
        self._freight_allowed: List[bool] = []
        self._kappa: List[float] = []
        self._I_min: List[float] = []
        self._I_max: List[float] = []

    def add_car_edge(self, u: int, v: int, kappa: float, I_min: float, I_max: float, freight_allowed: bool = True):
        self._edges.append((u, v))
        self._is_car.append(True)
        self._mode_of_edge.append(-1)
        self._freight_allowed.append(freight_allowed)
        self._kappa.append(kappa)
        self._I_min.append(I_min)
        self._I_max.append(I_max)

    def add_transit_edge(self, u: int, v: int, mode: int, kappa: float, I_min: float, I_max: float):
        if not (0 <= mode < self.n_modes_transit):
            raise ValueError(f"mode {mode} out of range [0, {self.n_modes_transit})")
        self._edges.append((u, v))
        self._is_car.append(False)
        self._mode_of_edge.append(mode)
        self._freight_allowed.append(False)
        self._kappa.append(kappa)
        self._I_min.append(I_min)
        self._I_max.append(I_max)

    def build(self) -> Network:
        net = Network(
            n_zones=self.n_zones,
            n_nodes=self.n_nodes,
            n_modes_transit=self.n_modes_transit,
            edges=list(self._edges),
            is_car=np.array(self._is_car, dtype=bool),
            mode_of_edge=np.array(self._mode_of_edge, dtype=int),
            freight_allowed=np.array(self._freight_allowed, dtype=bool),
            kappa=np.array(self._kappa, dtype=float),
            I_min=np.array(self._I_min, dtype=float),
            I_max=np.array(self._I_max, dtype=float),
            entry_node=self.entry_node,
        )
        return net
