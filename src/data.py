"""Enumerate (edge_seq, intermediate_nodes) and split into train/test."""
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .graph import Graph
from .config import TrainConfig


def enumerate_paths(graph: Graph, device: str = "cpu") -> dict:
    """Enumerate every path from the source through the graph.

    Assumes layer 0 has a single source node (id 0). Returns:
        edge_seqs: (N, L) long, with N = edges_per_node ** L, L = n_layers - 1
        nodes:     (N, n_layers) long, the full sequence of node ids visited
        labels:    (N,) long, nodes[:, -1]
    """
    assert graph.config.nodes_per_layer[0] == 1, "enumerate_paths assumes a single source"
    L = graph.n_layers - 1
    E = graph.edges_per_node
    grids = torch.meshgrid(*[torch.arange(E) for _ in range(L)], indexing="ij")
    edge_seqs = torch.stack(grids, dim=-1).reshape(-1, L)
    nodes = graph.traverse(edge_seqs.to(device)).cpu()
    return {"edge_seqs": edge_seqs, "nodes": nodes, "labels": nodes[:, -1]}


class PathDataset(Dataset):
    def __init__(self, edge_seqs: torch.Tensor, nodes: torch.Tensor):
        self.edge_seqs = edge_seqs
        self.nodes = nodes

    def __len__(self):
        return self.edge_seqs.shape[0]

    def __getitem__(self, idx):
        return {
            "edges": self.edge_seqs[idx],
            "nodes": self.nodes[idx],
            "label": self.nodes[idx, -1],
        }


def split_indices(n: int, train_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(round(train_frac * n))
    return perm[:n_train], perm[n_train:]


def build_dataloaders(graph: Graph, train_cfg: TrainConfig):
    paths = enumerate_paths(graph)
    dataset = PathDataset(paths["edge_seqs"], paths["nodes"])
    train_idx, test_idx = split_indices(len(dataset), train_cfg.train_frac, train_cfg.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=train_cfg.batch_size, shuffle=True, drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=train_cfg.batch_size, shuffle=False, drop_last=False,
    )
    return train_loader, test_loader, dataset, (train_idx, test_idx)
