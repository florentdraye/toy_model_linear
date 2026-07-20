"""Continue training from a saved checkpoint, optionally turning on new regularizers.

    python continue_train.py --from-ckpt runs/foo/ckpt.pt \
        --ridge-adv-lambda 1.0 --n-epochs 30 --out-dir runs/foo-ridge

Graph, model architecture, and train/test split are taken from the source ckpt.
All TrainConfig fields can be overridden via the CLI; anything not set keeps the
source's value. The optimizer is freshly initialized so cosine warmup restarts.

Typical use: load a linear-decodable checkpoint and turn on `--ridge-adv-lambda`
to study how the weights change as the model trades linearity for task accuracy.
"""
import argparse
from dataclasses import asdict, fields
from pathlib import Path

import torch

from src.config import GraphConfig, ModelConfig, TrainConfig
from src.graph import Graph
from src.train import train


def _add_train_overrides(p: argparse.ArgumentParser) -> None:
    """Every TrainConfig field exposed as a CLI flag with default=None."""
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--warmup-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval-every-epochs", type=int, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None,
                   choices=["adamw", "klshampoo"])
    p.add_argument("--ckpt-every-epochs", type=int, default=None)
    p.add_argument("--adv-probe-lambda", type=float, default=None)
    p.add_argument("--ortho-lambda", type=float, default=None)
    p.add_argument("--ridge-adv-lambda", type=float, default=None)
    p.add_argument("--ridge-adv-nu", type=float, default=None)
    p.add_argument("--ridge-adv-tau", type=float, default=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    _add_train_overrides(p)
    args = p.parse_args()

    ckpt_path = Path(args.from_ckpt)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    graph_cfg = GraphConfig(**ckpt["graph_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    src_train = dict(ckpt["train_cfg"])

    # Compose final TrainConfig: source ckpt as base, override only what user set on CLI.
    train_field_names = {f.name for f in fields(TrainConfig)}
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k in train_field_names
    }
    merged = {**src_train, **overrides}
    train_cfg = TrainConfig(**merged)

    graph_path = ckpt_path.parent / "graph.pt"
    graph = Graph.load(graph_path) if graph_path.exists() else Graph(graph_cfg)
    init_split = (ckpt["split"]["train"], ckpt["split"]["test"])

    print(f"[continue] from: {ckpt_path}")
    print(f"[continue] overrides: {overrides}")

    train(
        graph_cfg, model_cfg, train_cfg,
        out_dir=args.out_dir,
        init_state_dict=ckpt["model_state"],
        init_split=init_split,
        init_graph=graph,
    )

    # Append source-ckpt provenance to the new checkpoint so we can trace lineage.
    new_ckpt_path = Path(args.out_dir) / "ckpt.pt"
    new_ckpt = torch.load(new_ckpt_path, weights_only=False, map_location="cpu")
    new_ckpt["from_ckpt"] = str(ckpt_path.resolve())
    torch.save(new_ckpt, new_ckpt_path)
    print(f"[continue] tagged provenance -> {new_ckpt_path}  from={ckpt_path}")


if __name__ == "__main__":
    main()
