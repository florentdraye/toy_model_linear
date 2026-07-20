"""Linear probes for sampled latent node identities, at one or all graph layers.

For each target graph layer ell, sample `n_nodes` random node ids from that
layer. For each (transformer depth d, position p), fit a closed-form ridge
linear classifier with all `n_nodes` binary targets stacked. Report test AUC.

The expensive step is the single forward pass to collect hidden states. The
ridge factor (X^T X + lam I)^-1 is computed once per (d, p) and reused across
target layers, so probing all layers costs barely more than probing one.
"""
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn.functional as F

from .model import ToyTransformer


@dataclass
class SampledProbeConfig:
    n_nodes: int = 20
    middle_layer: int = -1            # -1 -> all non-trivial graph layers
    ridge_lam: float = 1e-3
    sample_seed: int = 0
    device: str = "cuda"


@torch.no_grad()
def collect_hidden_states(model: ToyTransformer, edges: torch.Tensor,
                          device: str, batch_size: int = 32768) -> torch.Tensor:
    """Returns (D, N, L, d_model) on `device`, fp32. D = n_blocks + 1."""
    model.eval()
    N = edges.shape[0]
    out = None
    for i in range(0, N, batch_size):
        chunk = edges[i:i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, h = model(chunk, return_hidden=True)
        h_stacked = torch.stack(h, dim=0).float()
        if out is None:
            D, _, L, d_model = h_stacked.shape
            out = torch.empty(D, N, L, d_model, device=device, dtype=torch.float32)
        out[:, i:i + chunk.shape[0]] = h_stacked
    return out


@torch.no_grad()
def binary_auc_columns(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUC per column (Mann-Whitney rank formula). scores, labels: (N, K)."""
    N, K = scores.shape
    aucs = torch.empty(K, device=scores.device)
    arange = torch.arange(1, N + 1, device=scores.device, dtype=torch.float64)
    for k in range(K):
        s, y = scores[:, k], labels[:, k]
        n_pos = y.sum()
        n_neg = N - n_pos
        if n_pos == 0 or n_neg == 0:
            aucs[k] = float("nan")
            continue
        ranks = torch.empty(N, device=s.device, dtype=torch.float64)
        ranks[s.argsort()] = arange
        pos_rank_sum = ranks[y.bool()].sum()
        aucs[k] = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return aucs


def run_sampled_probes(model: ToyTransformer,
                       edges: torch.Tensor, nodes: torch.Tensor,
                       train_idx: torch.Tensor, test_idx: torch.Tensor,
                       n_layers: int,
                       nodes_per_layer: List[int],
                       cfg: SampledProbeConfig) -> dict:
    """Probe each requested graph layer with `n_nodes` sampled latents."""
    device = cfg.device
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    nodes_dev = nodes.to(device)

    if cfg.middle_layer >= 0:
        target_layers = [cfg.middle_layer]
    else:
        target_layers = [ell for ell in range(n_layers) if nodes_per_layer[ell] > 1]

    # Sample from nodes that are actually reachable at each layer (layer 1 may
    # have far fewer reachable nodes than nodes_per_layer[1] = 100).
    rng = torch.Generator().manual_seed(cfg.sample_seed)
    chosen: Dict[int, torch.Tensor] = {}
    reachable_counts: Dict[int, int] = {}
    for ell in target_layers:
        reachable = nodes_dev[:, ell].unique()       # GPU
        reachable_counts[ell] = int(reachable.numel())
        if reachable.numel() <= cfg.n_nodes:
            chosen[ell] = reachable
        else:
            perm = torch.randperm(reachable.numel(), generator=rng)
            chosen[ell] = reachable[perm[:cfg.n_nodes].to(device)]

    Y_train, Y_test = {}, {}
    for ell in target_layers:
        c = chosen[ell]
        mid_tr = nodes_dev[train_idx, ell]
        mid_te = nodes_dev[test_idx, ell]
        Y_train[ell] = (mid_tr.unsqueeze(1) == c.unsqueeze(0)).float()
        Y_test[ell] = (mid_te.unsqueeze(1) == c.unsqueeze(0)).float()

    print(f"probing graph layers: {target_layers} (graph has {n_layers} layers)")
    for ell in target_layers:
        print(f"  layer {ell}: sampled {len(chosen[ell])} from "
              f"{reachable_counts[ell]} reachable (layer has {nodes_per_layer[ell]} nodes total)")
    print("collecting hidden states...", flush=True)
    hiddens = collect_hidden_states(model, edges, device)
    D, _, L, _ = hiddens.shape

    cubes: Dict[int, torch.Tensor] = {
        ell: torch.full((D, L, len(chosen[ell])), float("nan")) for ell in target_layers
    }

    for d in range(D):
        for p in range(L):
            feats = hiddens[d, :, p, :]
            X_train = feats[train_idx]
            X_test = feats[test_idx]
            ones_tr = torch.ones(X_train.shape[0], 1, device=device, dtype=X_train.dtype)
            ones_te = torch.ones(X_test.shape[0], 1, device=device, dtype=X_test.dtype)
            Xa_tr = torch.cat([X_train, ones_tr], dim=1)
            Xa_te = torch.cat([X_test, ones_te], dim=1)
            A = Xa_tr.T @ Xa_tr
            lam_eff = cfg.ridge_lam * A.diagonal().mean().clamp(min=1.0)
            A.diagonal().add_(lam_eff)
            LU, pivots = torch.linalg.lu_factor(A.double())  # shared across layers
            for ell in target_layers:
                B = Xa_tr.T @ Y_train[ell]
                W = torch.linalg.lu_solve(LU, pivots, B.double()).float()
                scores = Xa_te @ W
                cubes[ell][d, p] = binary_auc_columns(scores, Y_test[ell]).cpu()
        print(f"depth {d} done", flush=True)

    print_summary(cubes, target_layers, D, L)

    return {
        "cubes": {ell: cubes[ell] for ell in target_layers},
        "chosen": {ell: chosen[ell].cpu() for ell in target_layers},
        "target_layers": target_layers,
    }


def print_summary(cubes: Dict[int, torch.Tensor], target_layers: List[int],
                  D: int, L: int) -> None:
    for ell in target_layers:
        cube = cubes[ell]
        print(f"\n=== graph layer {ell}  (mean test AUC over {cube.shape[-1]} sampled reachable nodes) ===")
        header = "  depth\\pos  " + "  ".join(f"  p{p}" for p in range(L))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for d in range(D):
            row = "  ".join(f"{cube[d, p].mean().item():.3f}" for p in range(L))
            print(f"  d={d}        {row}")
