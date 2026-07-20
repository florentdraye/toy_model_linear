"""Entry point: linear probes for 20 sampled latents in the middle graph layer.

    python probe.py --ckpt runs/.../ckpt.pt
"""
import argparse
from pathlib import Path

import torch

from src.config import GraphConfig, ModelConfig
from src.graph import Graph
from src.model import ToyTransformer
from src.data import enumerate_paths
from src.probe import SampledProbeConfig, run_sampled_probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-nodes", type=int, default=20)
    p.add_argument("--middle-layer", type=int, default=-1,
                   help="-1 => scan all non-trivial graph layers; otherwise probe only that layer.")
    p.add_argument("--ridge-lam", type=float, default=1e-3)
    p.add_argument("--sample-seed", type=int, default=0)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, weights_only=False)
    graph_cfg = GraphConfig(**ckpt["graph_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])

    graph_path = ckpt_path.parent / "graph.pt"
    graph = Graph.load(graph_path) if graph_path.exists() else Graph(graph_cfg)

    model = ToyTransformer(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])

    paths = enumerate_paths(graph)
    train_idx = ckpt["split"]["train"]
    test_idx = ckpt["split"]["test"]

    cfg = SampledProbeConfig(
        n_nodes=args.n_nodes,
        middle_layer=args.middle_layer,
        ridge_lam=args.ridge_lam,
        sample_seed=args.sample_seed,
        device=args.device,
    )

    out = run_sampled_probes(
        model, paths["edge_seqs"], paths["nodes"],
        train_idx, test_idx,
        n_layers=graph_cfg.n_layers,
        nodes_per_layer=list(graph_cfg.nodes_per_layer),
        cfg=cfg,
    )

    save_path = ckpt_path.parent / "probes_sampled.pt"
    torch.save(out, save_path)
    print(f"\nsaved -> {save_path}")
    a_cube = next(iter(out["cubes"].values()))
    print(f"per-layer cube shape (D, L, n_nodes) = {tuple(a_cube.shape)}; "
          f"{len(out['target_layers'])} layers probed")


if __name__ == "__main__":
    main()
