"""Offline logistic-regression probes on the trained ckpt.

Uses the same probe subset (train_idx/test_idx + sample_seed subsampling) that
DoMProbeTracker would have used during training, so the resulting top-1s are
comparable to the linear-DoM top-1s already in dom_probe.pt.
"""
from pathlib import Path
import torch

from src.config import GraphConfig, ModelConfig
from src.graph import Graph
from src.data import enumerate_paths
from src.model import ToyTransformer
from src.dom_probe import DoMProbeConfig, DoMProbeTracker

RUN = Path("runs/20260720-105636-relu2-no-residual-fast")
device = "cuda"

ckpt = torch.load(RUN / "ckpt.pt", weights_only=False, map_location=device)
graph = Graph.load(RUN / "graph.pt")

model_cfg = ModelConfig(**ckpt["model_cfg"])
model = ToyTransformer(model_cfg).to(device).eval()
model.load_state_dict(ckpt["model_state"])

paths = enumerate_paths(graph, device=device)
edges, nodes = paths["edge_seqs"].to(device), paths["nodes"].to(device)
train_idx = ckpt["split"]["train"].to(device)
test_idx = ckpt["split"]["test"].to(device)

train_cfg_d = ckpt["train_cfg"]
probe_cfg = DoMProbeConfig(
    every_steps=train_cfg_d["dom_probe_every_steps"],
    n_nodes=train_cfg_d["dom_probe_n_nodes"],
    sample_seed=train_cfg_d["dom_probe_seed"],
    probe_max_train=train_cfg_d["dom_probe_max_train"],
    probe_max_test=train_cfg_d["dom_probe_max_test"],
    logreg_iters=300,          # <- enabled offline
    logreg_lr=3e-2,
)

tracker = DoMProbeTracker(
    edges=edges, nodes=nodes,
    train_idx=train_idx, test_idx=test_idx,
    n_layers=graph.config.n_layers,
    nodes_per_layer=list(graph.config.nodes_per_layer),
    cfg=probe_cfg,
)

with torch.no_grad():
    _, hiddens = model(tracker.edges, return_hidden=True)
hiddens = torch.stack(hiddens, dim=0)   # (D, B, L, d)
D, _, L, _ = hiddens.shape

print(f"{'layer':>5} {'reach':>5}  {'linDoM best (d,p) top1':>26}  {'logreg best (d,p) top1':>26}")
print("-" * 70)

# Also load the previously-computed linear DoM final top-1 for side-by-side.
dom = torch.load(RUN / "dom_probe.pt", weights_only=False)

for ell in tracker.target_layers:
    C = tracker.reachable_classes[ell].numel()

    lr_top1, lr_top5 = tracker._run_logreg_probes(hiddens, ell, D, L)
    flat = lr_top1.reshape(-1)
    b = int(flat.argmax())
    lr_bd, lr_bp = b // L, b % L
    lr_best = float(lr_top1[lr_bd, lr_bp])

    lin_final = dom["top1"][ell][-1]
    flat = lin_final.reshape(-1)
    b = int(flat.argmax())
    lin_bd, lin_bp = b // L, b % L
    lin_best = float(lin_final[lin_bd, lin_bp])

    print(f"{ell:>5} {C:>5}   ({lin_bd},{lin_bp}) {lin_best:6.3f}"
          f"           ({lr_bd},{lr_bp}) {lr_best:6.3f}")
