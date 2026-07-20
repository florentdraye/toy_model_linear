"""Post-hoc DoM + multinomial logistic regression probes at every (d, p) cell.

Loads a trained model from a ckpt, does one forward pass with return_hidden,
and for each graph layer with >1 reachable class prints two grids of top-1
test accuracy: DoM (mean-difference argmax) vs logistic regression (softmax
+ cross-entropy fit by Adam), plus the best cell per method.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import ModelConfig, GraphConfig
from src.graph import Graph
from src.data import enumerate_paths
from src.model import ToyTransformer
from src.dom_probe import _fit_batched_mlp


@torch.no_grad()
def collect_hiddens(model, edges, batch=8192):
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
def dom_top1_grid(hiddens, train_idx, test_idx, y_full, C):
    D, N, L, d = hiddens.shape
    y_tr = y_full[train_idx]; y_te = y_full[test_idx]
    Y_tr = F.one_hot(y_tr, C).float()
    n_pos = Y_tr.sum(0).clamp(min=1.0).unsqueeze(1)
    n_neg = (Y_tr.shape[0] - Y_tr.sum(0)).clamp(min=1.0).unsqueeze(1)
    top1 = torch.zeros(D, L)
    for dd in range(D):
        for p in range(L):
            X_tr = hiddens[dd, train_idx, p, :]
            X_te = hiddens[dd, test_idx, p, :]
            sum_pos = Y_tr.T @ X_tr
            sum_neg = X_tr.sum(0, keepdim=True) - sum_pos
            W = sum_pos / n_pos - sum_neg / n_neg
            pred = (X_te @ W.T).argmax(dim=1)
            top1[dd, p] = (pred == y_te).float().mean().cpu()
    return top1


def _fit_batched_logreg(X_tr, y_tr, X_te, y_te, n_iters, lr, C):
    """M multinomial logistic regressions in parallel via bmm."""
    M, Ntr, d = X_tr.shape
    device = X_tr.device
    X_tr = X_tr.detach(); X_te = X_te.detach()
    with torch.enable_grad():
        W = torch.zeros(M, d, C, device=device, requires_grad=True)
        b = torch.zeros(M, C, device=device, requires_grad=True)
        optim = torch.optim.Adam([W, b], lr=lr)
        y_tr_flat = y_tr.unsqueeze(0).expand(M, -1).reshape(-1)
        for _ in range(n_iters):
            logits = torch.bmm(X_tr, W) + b.unsqueeze(1)
            loss = F.cross_entropy(logits.reshape(-1, C), y_tr_flat)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    with torch.no_grad():
        pred = (torch.bmm(X_te, W) + b.unsqueeze(1)).argmax(dim=-1)
        return (pred == y_te.unsqueeze(0)).float().mean(dim=1).cpu()


def mlp_top1_grid(hiddens, train_idx, test_idx, y_full, C,
                  hidden, n_iters, lr, activation, cap):
    D, N, L, d = hiddens.shape
    tr_sub = train_idx[:min(cap, train_idx.numel())]
    X_tr = hiddens[:, tr_sub, :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d)
    X_te = hiddens[:, test_idx, :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d)
    top1, _ = _fit_batched_mlp(X_tr, y_full[tr_sub], X_te, y_full[test_idx],
                               hidden=hidden, n_iters=n_iters, lr=lr,
                               activation=activation, n_classes=C)
    return top1.view(D, L)


def logreg_top1_grid(hiddens, train_idx, test_idx, y_full, C,
                     n_iters, lr, cap):
    D, N, L, d = hiddens.shape
    tr_sub = train_idx[:min(cap, train_idx.numel())]
    X_tr = hiddens[:, tr_sub, :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d)
    X_te = hiddens[:, test_idx, :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d)
    top1 = _fit_batched_logreg(X_tr, y_full[tr_sub], X_te, y_full[test_idx],
                               n_iters, lr, C)
    return top1.view(D, L)


def fmt_grid(mat):
    D, L = mat.shape
    header = "  d\\p   " + "  ".join(f" p{p}" for p in range(L))
    lines = [header, "  " + "-" * (len(header) - 2)]
    for d in range(D):
        row = "  ".join(f"{mat[d, p].item():5.3f}" for p in range(L))
        lines.append(f"  d={d}   {row}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="ckpt.pt",
                    help="Ckpt filename inside --run (e.g. ckpt_step1000.pt).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument("--logreg-max-train", type=int, default=10000)
    ap.add_argument("--logreg-iters", type=int, default=300)
    ap.add_argument("--logreg-lr", type=float, default=3e-2)
    ap.add_argument("--mlp-hidden", type=int, default=0,
                    help="Hidden dim for MLP probes; 0 disables.")
    ap.add_argument("--mlp-iters", type=int, default=500)
    ap.add_argument("--mlp-lr", type=float, default=1e-2)
    ap.add_argument("--mlp-max-train", type=int, default=15000)
    ap.add_argument("--mlp-activation", type=str, default="relu",
                    choices=["relu", "gelu"])
    args = ap.parse_args()

    ckpt = torch.load(Path(args.run) / args.ckpt, weights_only=False,
                      map_location=args.device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    graph_cfg = GraphConfig(**ckpt["graph_cfg"])
    graph = Graph.load(Path(args.run) / "graph.pt")
    paths = enumerate_paths(graph)
    edges_all = paths["edge_seqs"].to(args.device)
    nodes_all = paths["nodes"].to(args.device)
    train_idx = ckpt["split"]["train"].to(args.device)
    test_idx = ckpt["split"]["test"].to(args.device)
    if train_idx.numel() > args.max_train:
        train_idx = train_idx[:args.max_train]
    if test_idx.numel() > args.max_test:
        test_idx = test_idx[:args.max_test]

    model = ToyTransformer(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_idx = torch.cat([train_idx, test_idx])
    edges_sub = edges_all[all_idx]
    nodes_sub = nodes_all[all_idx]
    n_tr = train_idx.numel()
    train_range = torch.arange(n_tr, device=args.device)
    test_range = torch.arange(n_tr, n_tr + test_idx.numel(), device=args.device)

    hiddens = collect_hiddens(model, edges_sub)
    D, _, L, d = hiddens.shape
    print(f"model: n_blocks={model_cfg.n_blocks}  seq_len={model_cfg.seq_len}  "
          f"d_model={model_cfg.d_model}  d_ff={model_cfg.d_ff}")
    print(f"train/test used: {n_tr} / {test_idx.numel()}\n")

    target_layers = [ell for ell in range(graph_cfg.n_layers)
                     if graph_cfg.nodes_per_layer[ell] > 1]

    for ell in target_layers:
        reachable = nodes_sub[:, ell].unique()
        C = reachable.numel()
        remap = torch.full((int(reachable.max().item()) + 1,), -1,
                           dtype=torch.long, device=args.device)
        remap[reachable] = torch.arange(C, device=args.device)
        y_full = remap[nodes_sub[:, ell]]

        dom = dom_top1_grid(hiddens, train_range, test_range, y_full, C)
        lr_grid = logreg_top1_grid(hiddens, train_range, test_range, y_full, C,
                                   args.logreg_iters, args.logreg_lr,
                                   args.logreg_max_train)
        dbi = dom.flatten().argmax().item()
        lbi = lr_grid.flatten().argmax().item()

        print(f"=== graph layer {ell}   (C={C} reachable classes, "
              f"chance={1.0/C:.3f}) ===")
        print("DoM top-1:")
        print(fmt_grid(dom))
        print(f"best DoM    cell: (d={dbi//L}, p={dbi%L})  top-1={dom.flatten()[dbi]:.3f}")
        print()
        print(f"LogReg top-1 (iters={args.logreg_iters}):")
        print(fmt_grid(lr_grid))
        print(f"best LogReg cell: (d={lbi//L}, p={lbi%L})  top-1={lr_grid.flatten()[lbi]:.3f}")
        print(f"peak uplift LogReg - DoM: "
              f"{lr_grid.flatten()[lbi].item() - dom.flatten()[dbi].item():+.3f}")
        if args.mlp_hidden > 0:
            mlp = mlp_top1_grid(hiddens, train_range, test_range, y_full, C,
                                args.mlp_hidden, args.mlp_iters, args.mlp_lr,
                                args.mlp_activation, args.mlp_max_train)
            mbi = mlp.flatten().argmax().item()
            print()
            print(f"MLP top-1 (hidden={args.mlp_hidden}, iters={args.mlp_iters}):")
            print(fmt_grid(mlp))
            print(f"best MLP    cell: (d={mbi//L}, p={mbi%L})  top-1={mlp.flatten()[mbi]:.3f}")
            print(f"peak uplift MLP    - LogReg: "
                  f"{mlp.flatten()[mbi].item() - lr_grid.flatten()[lbi].item():+.3f}")
        print()


if __name__ == "__main__":
    main()
