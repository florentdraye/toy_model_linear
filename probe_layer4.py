"""Difference-of-means multi-class probe for the final graph layer (layer 4).

For each (transformer depth d, position p) in the residual stream, fit a DoM
direction w_c = mean(X | y=c) - mean(X | y!=c) for every class c on the train
split, then predict argmax over classes on the test split. Reports top-1 and
top-5 accuracy per (d, p), highlights the best cell.

Compared against the model's own head test_acc as a reference.
"""
import argparse
from pathlib import Path

import torch

from src.config import ModelConfig, GraphConfig
from src.graph import Graph
from src.data import enumerate_paths
from src.model import ToyTransformer


@torch.no_grad()
def collect_hiddens(model, edges, batch=16384):
    N = edges.shape[0]
    out = None
    for i in range(0, N, batch):
        chunk = edges[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, h = model(chunk, return_hidden=True)
        hs = torch.stack(h, dim=0).float()
        if out is None:
            D, _, L, d = hs.shape
            out = torch.empty(D, N, L, d, device=edges.device, dtype=torch.float32)
        out[:, i:i + chunk.shape[0]] = hs
    return out


@torch.no_grad()
def dom_directions(X_tr, y_tr, C):
    """Build (C, d) DoM directions. w_c = mean(X | y==c) - mean(X | y!=c)."""
    Y = torch.nn.functional.one_hot(y_tr, C).float()   # (Ntr, C)
    n_pos = Y.sum(0).clamp(min=1.0)                    # (C,)
    n_neg = (Y.shape[0] - Y.sum(0)).clamp(min=1.0)
    sum_pos = Y.T @ X_tr                                # (C, d)
    sum_all = X_tr.sum(0, keepdim=True)                 # (1, d)
    sum_neg = sum_all - sum_pos
    return sum_pos / n_pos.unsqueeze(1) - sum_neg / n_neg.unsqueeze(1)


def fmt_grid(mat, row_label="d", cell="{:5.3f}"):
    D, L = mat.shape
    header = f"  {row_label}\\p  " + " ".join(f"  p{p}" for p in range(L))
    lines = [header, "  " + "-" * (len(header) - 2)]
    for d in range(D):
        row = " ".join(cell.format(mat[d, p].item()) for p in range(L))
        lines.append(f"  d={d}   {row}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/dom_analysis")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--target-layer", type=int, default=-1,
                    help="Graph layer to probe; -1 -> last layer.")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.run) / "ckpt.pt", weights_only=False,
                      map_location=args.device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    graph_cfg = GraphConfig(**ckpt["graph_cfg"])

    graph = Graph.load(Path(args.run) / "graph.pt")
    paths = enumerate_paths(graph)
    edges_all = paths["edge_seqs"].to(args.device)
    nodes_all = paths["nodes"].to(args.device)

    train_idx = ckpt["split"]["train"].to(args.device)
    test_idx = ckpt["split"]["test"].to(args.device)

    model = ToyTransformer(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_layer = graph_cfg.n_layers - 1 if args.target_layer < 0 else args.target_layer
    y = nodes_all[:, target_layer]
    C = int(y.max().item()) + 1                          # active class count
    y_tr, y_te = y[train_idx], y[test_idx]

    print(f"target graph layer: {target_layer}  (num classes reachable: {C})")
    print(f"train / test: {y_tr.numel()} / {y_te.numel()}")

    hiddens = collect_hiddens(model, edges_all)
    D, N, L, d = hiddens.shape
    print(f"hidden cube (D, N, L, d): {(D, N, L, d)}\n")

    top1 = torch.zeros(D, L)
    top5 = torch.zeros(D, L)
    for depth in range(D):
        for pos in range(L):
            X_tr = hiddens[depth, train_idx, pos, :]
            X_te = hiddens[depth, test_idx, pos, :]
            W = dom_directions(X_tr, y_tr, C)             # (C, d)
            scores = X_te @ W.T                            # (Nte, C)
            pred1 = scores.argmax(dim=1)
            top1[depth, pos] = (pred1 == y_te).float().mean().cpu()
            top5_pred = scores.topk(5, dim=1).indices
            top5[depth, pos] = (top5_pred == y_te.unsqueeze(1)).any(1).float().mean().cpu()

    # Reference: the model's own final-head test accuracy.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits_te = model(edges_all[test_idx])
    head_top1 = (logits_te.argmax(dim=1) == y_te).float().mean().item()
    head_top5 = (logits_te.topk(5, dim=1).indices == y_te.unsqueeze(1)).any(1).float().mean().item()

    print("DoM multi-class TOP-1 test accuracy per (depth, position):")
    print(fmt_grid(top1))
    print("\nDoM multi-class TOP-5 test accuracy per (depth, position):")
    print(fmt_grid(top5))

    flat = top1.flatten()
    best = flat.argmax().item()
    bd, bp = best // L, best % L
    print(f"\nchance top-1: {1.0 / C:.3f}   chance top-5: {5.0 / C:.3f}")
    print(f"model head  top-1: {head_top1:.3f}   top-5: {head_top5:.3f}")
    print(f"best DoM cell (d={bd}, p={bp}): "
          f"top-1 {top1[bd, bp].item():.3f}   top-5 {top5[bd, bp].item():.3f}")


if __name__ == "__main__":
    main()
