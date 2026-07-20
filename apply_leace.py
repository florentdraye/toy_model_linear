"""Apply LEACE concept erasure at one residual-stream depth and report effects.

    python apply_leace.py --ckpt runs/.../ckpt.pt --graph-layer 3 --depth 2

Fits a per-position LEACE eraser on the training residuals at the chosen depth,
using graph-layer-`l` node identity as the categorical concept. Installs a
forward hook so the rest of the model sees the erased residual. Reports:
  - test accuracy pre vs post LEACE
  - full probe cube under the post-LEACE forward pass (probes at the erased
    depth should be at chance for layer `l`; downstream depths show whether the
    model has any chance of reconstructing the latent post-intervention).
"""
import argparse
from pathlib import Path

import torch

from src.config import GraphConfig, ModelConfig
from src.graph import Graph
from src.model import ToyTransformer
from src.data import enumerate_paths
from src.leace import LeaceEraser
from src.probe import SampledProbeConfig, run_sampled_probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--graph-layer", type=int, default=3,
                   help="Graph layer whose node identity is the concept to erase.")
    p.add_argument("--depth", type=int, default=2,
                   help="Residual-stream depth at which to apply LEACE (1..n_blocks). "
                        "After block (depth-1)'s output.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip running the post-LEACE probe cube (acc-only mode).")
    p.add_argument("--sigma-reg", type=float, default=0.0,
                   help="Ridge added to Σ_XX during LEACE fit.")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, weights_only=False)
    graph_cfg = GraphConfig(**ckpt["graph_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    assert 1 <= args.depth <= model_cfg.n_blocks, (
        f"--depth must be in [1, n_blocks={model_cfg.n_blocks}]"
    )
    assert 0 <= args.graph_layer < graph_cfg.n_layers, (
        f"--graph-layer must be in [0, n_layers={graph_cfg.n_layers})"
    )

    graph_path = ckpt_path.parent / "graph.pt"
    graph = Graph.load(graph_path) if graph_path.exists() else Graph(graph_cfg)

    model = ToyTransformer(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    paths = enumerate_paths(graph)
    edges_all = paths["edge_seqs"].to(args.device)
    nodes_all = paths["nodes"].to(args.device)
    labels_all = nodes_all[:, -1]
    train_idx = ckpt["split"]["train"].to(args.device)
    test_idx = ckpt["split"]["test"].to(args.device)
    edges_train = edges_all[train_idx]
    nodes_train = nodes_all[train_idx]
    edges_test = edges_all[test_idx]
    labels_test = labels_all[test_idx]

    # Collect residuals at the chosen depth on the training set (batched; no hook yet).
    N_train = edges_train.shape[0]
    chunk = 32768
    h_d = None
    with torch.no_grad():
        for i in range(0, N_train, chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, hiddens = model(edges_train[i:i + chunk], return_hidden=True)
            h_chunk = hiddens[args.depth].float()
            if h_d is None:
                h_d = torch.empty(N_train, h_chunk.shape[1], h_chunk.shape[2],
                                  device=args.device, dtype=torch.float32)
            h_d[i:i + h_chunk.shape[0]] = h_chunk
    # h_d: (N_train, L, D)
    z_train = nodes_train[:, args.graph_layer]        # (N_train,) layer-l node id
    L = h_d.shape[1]
    K = int(z_train.max().item()) + 1

    print(f"fitting LEACE: depth={args.depth}, concept=layer{args.graph_layer} "
          f"(K={K} classes), L={L} positions, N={h_d.shape[0]}, D={h_d.shape[-1]}")
    erasers = []
    for p_idx in range(L):
        e = LeaceEraser.fit(h_d[:, p_idx, :], z_train, sigma_reg=args.sigma_reg)
        erasers.append(e.to(args.device))
        print(f"  pos {p_idx}: rank={e.rank}")

    def batched_acc(edges, labels, batch=32768):
        correct = 0
        n = edges.shape[0]
        with torch.no_grad():
            for i in range(0, n, batch):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(edges[i:i + batch])
                correct += (logits.argmax(-1) == labels[i:i + batch]).sum().item()
        return correct / n

    acc_pre = batched_acc(edges_test, labels_test)
    print(f"\npre-LEACE test acc:  {acc_pre:.4f}")

    # Hook on output of block[depth-1], so hiddens[depth] = LEACE(block[depth-1](.)).
    # Run the erasure in fp32 so bf16-autocast noise doesn't leak the concept.
    def hook(module, inputs, output):
        h = output                                       # bf16 under autocast
        h32 = h.float()
        outs = [erasers[p_idx](h32[:, p_idx, :]) for p_idx in range(L)]
        return torch.stack(outs, dim=1).to(h.dtype)

    handle = model.blocks[args.depth - 1].register_forward_hook(hook)
    try:
        acc_post = batched_acc(edges_test, labels_test)
        print(f"post-LEACE test acc: {acc_post:.4f}")
        print(f"Δ acc:               {acc_post - acc_pre:+.4f}")

        save_payload = {
            "graph_layer": args.graph_layer,
            "depth": args.depth,
            "acc_pre": acc_pre,
            "acc_post": acc_post,
            "ranks": [e.rank for e in erasers],
            "erasers_cpu": [(e.mu.cpu(), e.B.cpu(), e.Bp.cpu()) for e in erasers],
        }

        if not args.no_probe:
            print("\nrunning probe cube under LEACE intervention...", flush=True)
            cfg = SampledProbeConfig(
                n_nodes=20, middle_layer=-1, ridge_lam=1e-3,
                sample_seed=0, device=args.device,
            )
            out = run_sampled_probes(
                model, paths["edge_seqs"], paths["nodes"],
                ckpt["split"]["train"], ckpt["split"]["test"],
                n_layers=graph_cfg.n_layers,
                nodes_per_layer=list(graph_cfg.nodes_per_layer),
                cfg=cfg,
            )
            save_payload["probes_post"] = out
    finally:
        handle.remove()

    save = ckpt_path.parent / f"leace_layer{args.graph_layer}_d{args.depth}.pt"
    torch.save(save_payload, save)
    print(f"\nsaved -> {save}")


if __name__ == "__main__":
    main()
