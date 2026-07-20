"""Random layered DAG.

Layer 0 has nodes_per_layer[0] nodes (typically 1 = the source). Layer i for
i > 0 has nodes_per_layer[i] nodes. Each node at layer i (i < n_layers - 1)
has `edges_per_node` outgoing edges, each pointing to a *distinct* node in
layer i+1, sampled uniformly without replacement.

Edges are *locally* indexed: edge index e from node s at layer i deterministi-
cally leads to some node in layer i+1, given by `edges[i][s, e]`.
"""
from pathlib import Path
from dataclasses import asdict

import torch

from .config import GraphConfig


class Graph:
    def __init__(self, config: GraphConfig):
        self.config = config
        rng = torch.Generator().manual_seed(config.seed)
        self.edges = []  # list[Tensor (n_src, E) long]; edges[i][s, e] -> node id in layer i+1
        for i in range(config.n_layers - 1):
            n_src = config.nodes_per_layer[i]
            n_dst = config.nodes_per_layer[i + 1]
            E = config.edges_per_node
            edges_i = torch.empty(n_src, E, dtype=torch.long)
            for s in range(n_src):
                edges_i[s] = torch.randperm(n_dst, generator=rng)[:E]
            self.edges.append(edges_i)

    @property
    def n_layers(self):
        return self.config.n_layers

    @property
    def edges_per_node(self):
        return self.config.edges_per_node

    def traverse(self, edge_seq: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Walk the graph following local edge indices.

        Args:
            edge_seq: (B, n_layers - 1) long tensor of local edge indices.
            start: starting node id in layer 0 (default 0).

        Returns:
            (B, n_layers) long tensor of node ids visited at each layer
            (including the source and the final node).
        """
        B = edge_seq.shape[0]
        device = edge_seq.device
        current = torch.full((B,), start, dtype=torch.long, device=device)
        nodes = [current]
        for i, e_layer in enumerate(self.edges):
            e_layer = e_layer.to(device)
            current = e_layer[current, edge_seq[:, i]]
            nodes.append(current)
        return torch.stack(nodes, dim=1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config),
                    "edges": [e.cpu() for e in self.edges]}, path)

    @classmethod
    def load(cls, path) -> "Graph":
        blob = torch.load(path, weights_only=False)
        cfg = GraphConfig(**blob["config"])
        g = cls.__new__(cls)
        g.config = cfg
        g.edges = blob["edges"]
        return g
