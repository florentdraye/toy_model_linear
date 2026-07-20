"""Do MLP neurons correlate with graph nodes?

For a trained checkpoint with d_ff = nodes_per_layer (here 100), each block
has a 100-dim MLP hidden state. We ask: does each neuron fire selectively
for one specific graph node visited along the path?

For each block d we:
  1. Forward the test set, capture post-activation MLP hidden h_d \\in R^{B x L x d_ff}
     (we use the *last* position p=L-1, since the head reads only that one).
  2. For each graph layer ell in [1..n_layers-1], build the binary target
     matrix Y_ell of shape (B, n_nodes_ell) = (test_nodes[:, ell] == node_id).
  3. Compute AUC of every neuron vs every (ell, node) using the rank formula.
     Per neuron: sort once, dot-product the rank vector with each binary column.
  4. Report: for each (block, graph_layer), the assignment quality via the
     row maxima of the (neuron x node) AUC matrix, plus an injective matching
     score (sum of best-AUC under one-to-one Hungarian on `linear_sum_assignment`
     if scipy is available, else a greedy fallback).

Usage:
    python analyze_neurons.py runs/20260626-163507-dff100/ckpt.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import GraphConfig, ModelConfig
from src.graph import Graph
from src.model import ToyTransformer
from src.data import enumerate_paths


@torch.no_grad()
def collect_mlp_hidden(model: ToyTransformer, edges: torch.Tensor,
                       device: str, position: int,
                       batch_size: int = 16384) -> torch.Tensor:
    """Returns (n_blocks, N, d_ff) post-activation MLP hidden at the given position."""
    model.eval()
    n_blocks = len(model.blocks)
    d_ff = model.cfg.d_ff
    N = edges.shape[0]
    out = torch.empty(n_blocks, N, d_ff, device=device, dtype=torch.float32)

    captured: List[torch.Tensor] = [None] * n_blocks
    hooks = []
    def make_hook(b_idx):
        def _hook(_mod, _inp, output):
            captured[b_idx] = output
        return _hook
    for b, block in enumerate(model.blocks):
        # block.mlp = Sequential(Linear, act, Linear); index 1 is the activation
        hooks.append(block.mlp[1].register_forward_hook(make_hook(b)))

    try:
        for i in range(0, N, batch_size):
            chunk = edges[i:i + batch_size].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model(chunk)
            for b in range(n_blocks):
                # captured[b] has shape (B, L, d_ff)
                out[b, i:i + chunk.shape[0]] = captured[b][:, position, :].float()
    finally:
        for h in hooks:
            h.remove()
    return out


@torch.no_grad()
def auc_matrix(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUC of every score column vs every label column.

    scores: (N, K_s) float
    labels: (N, K_l) float in {0, 1}
    returns: (K_s, K_l) AUC
    Uses Mann-Whitney rank formula. Cost: K_s sorts + one (K_s, K_l) matmul.
    """
    N, K_s = scores.shape
    _, K_l = labels.shape
    device = scores.device
    n_pos = labels.sum(dim=0)                                # (K_l,)
    n_neg = N - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    arange = torch.arange(1, N + 1, device=device, dtype=torch.float32)
    out = torch.full((K_s, K_l), float("nan"), device=device)
    for k in range(K_s):
        order = scores[:, k].argsort()
        ranks = torch.empty(N, device=device, dtype=torch.float32)
        ranks[order] = arange
        pos_rank_sum = ranks @ labels                        # (K_l,)
        denom = n_pos * n_neg
        auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / denom.clamp(min=1)
        auc[~valid] = float("nan")
        out[k] = auc
    return out


def greedy_assignment(aucs: torch.Tensor) -> Tuple[List[Tuple[int, int]], float]:
    """Greedy one-to-one matching: repeatedly pick the highest remaining cell."""
    A = aucs.clone()
    # NaN means unreachable; treat as -inf for matching
    A[torch.isnan(A)] = -1.0
    R, C = A.shape
    used_r = torch.zeros(R, dtype=torch.bool)
    used_c = torch.zeros(C, dtype=torch.bool)
    pairs: List[Tuple[int, int]] = []
    total = 0.0
    K = min(R, C)
    for _ in range(K):
        flat = A.flatten().argmax().item()
        r, c = divmod(flat, C)
        if used_r[r] or used_c[c]:
            break  # shouldn't happen because we set rows/cols to -inf
        pairs.append((r, c))
        total += A[r, c].item()
        A[r, :] = -1.0
        A[:, c] = -1.0
        used_r[r] = True
        used_c[c] = True
    return pairs, total / max(1, K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--position", type=int, default=-1,
                   help="Token position to read MLP hidden from. -1 = last.")
    p.add_argument("--max-test", type=int, default=200_000,
                   help="Cap test samples for AUC computation.")
    p.add_argument("--save", type=str, default="",
                   help="Optional path to save full AUC tensors.")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    blob = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model_cfg = ModelConfig(**blob["model_cfg"])
    graph_cfg = GraphConfig(**blob["graph_cfg"])

    print(f"loaded {ckpt_path}")
    print(f"  d_model={model_cfg.d_model}  d_ff={model_cfg.d_ff}  "
          f"n_blocks={model_cfg.n_blocks}  n_heads={model_cfg.n_heads}  "
          f"act={model_cfg.mlp_activation}")
    print(f"  graph: n_layers={graph_cfg.n_layers}  "
          f"nodes_per_layer={graph_cfg.nodes_per_layer}  "
          f"edges_per_node={graph_cfg.edges_per_node}")

    # Reproduce dataset
    graph_path = ckpt_path.parent / "graph.pt"
    graph = Graph.load(graph_path) if graph_path.exists() else Graph(graph_cfg)
    paths = enumerate_paths(graph)
    edges_all = paths["edge_seqs"]
    nodes_all = paths["nodes"]
    test_idx = blob["split"]["test"]
    if test_idx.numel() > args.max_test:
        test_idx = test_idx[:args.max_test]
    edges_te = edges_all[test_idx]
    nodes_te = nodes_all[test_idx]

    model = ToyTransformer(model_cfg).to(args.device)
    model.load_state_dict(blob["model_state"])

    L = model_cfg.seq_len
    pos = args.position if args.position >= 0 else L - 1
    print(f"\nreading MLP hidden at position {pos} (seq_len={L})")
    print(f"test samples: {edges_te.shape[0]:,}")

    print("collecting MLP hidden states...", flush=True)
    hidden = collect_mlp_hidden(model, edges_te, args.device, pos)   # (n_blocks, N, d_ff)
    n_blocks, N, d_ff = hidden.shape
    print(f"hidden shape: {tuple(hidden.shape)}")

    nodes_te = nodes_te.to(args.device)
    n_layers = graph_cfg.n_layers
    target_layers = [ell for ell in range(n_layers) if graph_cfg.nodes_per_layer[ell] > 1]

    # All AUC tensors: per block, per graph_layer -> (d_ff, n_nodes_ell)
    all_aucs: Dict[int, Dict[int, torch.Tensor]] = {b: {} for b in range(n_blocks)}

    for ell in target_layers:
        n_ell = graph_cfg.nodes_per_layer[ell]
        node_ids = torch.arange(n_ell, device=args.device)
        Y = (nodes_te[:, ell].unsqueeze(1) == node_ids.unsqueeze(0)).float()  # (N, n_ell)
        for b in range(n_blocks):
            scores = hidden[b]
            A = auc_matrix(scores, Y)   # (d_ff, n_ell)
            all_aucs[b][ell] = A.cpu()
        print(f"  ell={ell}: computed AUCs for all blocks", flush=True)

    # ---- Reporting -------------------------------------------------------
    print("\n" + "=" * 72)
    print("Per-neuron best AUC over nodes, summarized per (block, graph_layer)")
    print("=" * 72)
    print(f"{'block':>5} {'gL':>3} {'reach':>6} | "
          f"{'mean_max':>9} {'med_max':>9} {'n>=0.99':>7} {'n>=0.95':>7} {'n>=0.90':>7} | "
          f"{'match_mean':>10}")
    for b in range(n_blocks):
        for ell in target_layers:
            A = all_aucs[b][ell]
            reach = int((~torch.isnan(A[0])).sum().item())
            # row-max ignoring NaN
            Asafe = torch.where(torch.isnan(A), torch.tensor(-1.0), A)
            row_max, _ = Asafe.max(dim=1)
            valid = row_max > -1
            rm = row_max[valid]
            mean_max = rm.mean().item()
            med_max = rm.median().item()
            n99 = int((rm >= 0.99).sum().item())
            n95 = int((rm >= 0.95).sum().item())
            n90 = int((rm >= 0.90).sum().item())
            # Hungarian-ish: greedy injective matching over reachable nodes
            # Restrict columns to reachable
            mask = ~torch.isnan(A[0])
            A_reach = A[:, mask]
            _, match_score = greedy_assignment(A_reach)
            print(f"{b:>5} {ell:>3} {reach:>6} | "
                  f"{mean_max:>9.3f} {med_max:>9.3f} {n99:>7d} {n95:>7d} {n90:>7d} | "
                  f"{match_score:>10.3f}")
        print()

    # ---- Which graph layer does each block "prefer"? --------------------
    print("=" * 72)
    print("For each block, which graph layer maximizes per-neuron alignment?")
    print("(mean over neurons of [max AUC across nodes in that layer])")
    print("=" * 72)
    header = "block | " + " ".join(f"  ell={ell}" for ell in target_layers)
    print(header)
    print("-" * len(header))
    for b in range(n_blocks):
        cells = []
        for ell in target_layers:
            A = all_aucs[b][ell]
            Asafe = torch.where(torch.isnan(A), torch.tensor(-1.0), A)
            row_max, _ = Asafe.max(dim=1)
            valid = row_max > -1
            cells.append(f"{row_max[valid].mean().item():>7.3f}")
        print(f"{b:>5} | " + " ".join(cells))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"aucs": all_aucs, "target_layers": target_layers,
                    "position": pos, "ckpt": str(ckpt_path)}, save_path)
        print(f"\nsaved AUC tensors -> {save_path}")


if __name__ == "__main__":
    main()
