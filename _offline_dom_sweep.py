"""Sweep DoM sample budget: is layer 6's linear-DoM underperformance a
statistical-estimator problem (not enough samples per class to nail the
class-conditional mean), or genuinely a lack of linear signal?
"""
from pathlib import Path
import torch

from src.config import ModelConfig
from src.graph import Graph
from src.data import enumerate_paths
from src.model import ToyTransformer
from src.dom_probe import DoMProbeConfig, DoMProbeTracker

RUN = Path("runs/20260720-105636-relu2-no-residual-fast")
device = "cuda"

ckpt = torch.load(RUN / "ckpt.pt", weights_only=False, map_location=device)
graph = Graph.load(RUN / "graph.pt")
model = ToyTransformer(ModelConfig(**ckpt["model_cfg"])).to(device).eval()
model.load_state_dict(ckpt["model_state"])

paths = enumerate_paths(graph, device=device)
edges, nodes = paths["edge_seqs"].to(device), paths["nodes"].to(device)
train_idx = ckpt["split"]["train"].to(device)
test_idx = ckpt["split"]["test"].to(device)


def linear_dom_top1(probe_max_train: int, probe_max_test: int = 10000):
    cfg = DoMProbeConfig(
        every_steps=1, n_nodes=10, sample_seed=0,
        probe_max_train=probe_max_train, probe_max_test=probe_max_test,
    )
    tr = DoMProbeTracker(
        edges=edges, nodes=nodes,
        train_idx=train_idx, test_idx=test_idx,
        n_layers=graph.config.n_layers,
        nodes_per_layer=list(graph.config.nodes_per_layer),
        cfg=cfg,
    )
    # Forward pass in chunks to stay under VRAM.
    with torch.no_grad():
        B = tr.edges.shape[0]
        chunks = []
        for s in range(0, B, 20000):
            _, h = model(tr.edges[s:s+20000], return_hidden=True)
            chunks.append(torch.stack(h, dim=0))   # (D, b, L, d)
        hiddens = torch.cat(chunks, dim=1)   # (D, B, L, d)
    D, _, L, d = hiddens.shape

    out = {}
    for ell in tr.target_layers:
        C = tr.reachable_classes[ell].numel()
        y_tr = tr.y_train_full[ell]
        y_te = tr.y_test_full[ell]
        Y_tr_mc = tr.Y_train_mc[ell]
        n_pos = tr.n_pos_mc[ell]
        n_neg = tr.n_neg_mc[ell]
        X_tr_all = hiddens[:, tr.train_idx, :, :]   # (D, Ntr, L, d)
        X_te_all = hiddens[:, tr.test_idx,  :, :]   # (D, Nte, L, d)
        best = -1.0
        best_dp = (0, 0)
        for dd in range(D):
            for pp in range(L):
                X_tr = X_tr_all[dd, :, pp, :]
                X_te = X_te_all[dd, :, pp, :]
                sum_all = X_tr.sum(dim=0)
                sum_pos = Y_tr_mc.T @ X_tr
                sum_neg = sum_all - sum_pos
                W = sum_pos / n_pos.unsqueeze(1) - sum_neg / n_neg.unsqueeze(1)
                scores = X_te @ W.T
                top1 = (scores.argmax(dim=1) == y_te).float().mean().item()
                if top1 > best:
                    best = top1
                    best_dp = (dd, pp)
        out[ell] = (best, best_dp, C)
    return out, tr.train_idx.numel(), tr.test_idx.numel()


header_layers = list(range(1, 7))
print(f"{'probe_max_train':>16}  {'Ntr':>7}  {'Nte':>6}  " + " ".join(f"L{ell:>10}" for ell in header_layers))
for budget in [10000, 30000, 100000, 300000, 0]:   # 0 = all
    out, Ntr, Nte = linear_dom_top1(budget)
    row = f"{budget:>16}  {Ntr:>7}  {Nte:>6}  "
    for ell in header_layers:
        top1, (dd, pp), C = out[ell]
        row += f"({dd},{pp}){top1:6.3f}  "
    print(row)
