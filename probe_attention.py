"""Evaluate DoM decodability immediately after a transformer's attention sublayer.

This loads an existing checkpoint; it does not retrain the model.  It reports
the requested post-attention cell alongside the post-full-block cell used by
the standard residual-depth plots.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import GraphConfig, ModelConfig
from src.data import enumerate_paths
from src.graph import Graph
from src.model import ToyTransformer


def _dom_top1(x_train, x_test, y_train_raw, y_test_raw):
    classes = torch.cat([y_train_raw, y_test_raw]).unique()
    remap = torch.full(
        (int(classes.max().item()) + 1,), -1, dtype=torch.long,
        device=y_train_raw.device,
    )
    remap[classes] = torch.arange(classes.numel(), device=y_train_raw.device)
    y_train = remap[y_train_raw]
    y_test = remap[y_test_raw]
    targets = F.one_hot(y_train, classes.numel()).float()
    n_pos = targets.sum(0).clamp(min=1).unsqueeze(1)
    n_neg = (targets.shape[0] - targets.sum(0)).clamp(min=1).unsqueeze(1)
    sum_all = x_train.sum(0, keepdim=True)
    sum_pos = targets.T @ x_train
    directions = sum_pos / n_pos - (sum_all - sum_pos) / n_neg
    predictions = (x_test @ directions.T).argmax(1)
    return (predictions == y_test).float().mean().item(), classes.numel()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--block", type=int, default=1,
                        help="One-based transformer block number.")
    parser.add_argument("--graph-layer", type=int, default=3)
    parser.add_argument("--position", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu",
                            weights_only=False)
    model_cfg = ModelConfig(**checkpoint["model_cfg"])
    graph_cfg = GraphConfig(**checkpoint["graph_cfg"])
    train_cfg = checkpoint["train_cfg"]
    block_index = args.block - 1
    if not 0 <= block_index < model_cfg.n_blocks:
        raise ValueError(f"--block must be in 1..{model_cfg.n_blocks}")
    if not 0 <= args.position < model_cfg.seq_len:
        raise ValueError(f"--position must be in 0..{model_cfg.seq_len - 1}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    graph_path = out_dir / "graph.pt"
    graph = Graph.load(graph_path) if graph_path.exists() else Graph(graph_cfg)
    paths = enumerate_paths(graph)
    train_idx = checkpoint["split"]["train"]
    test_idx = checkpoint["split"]["test"]

    # Match the DoM tracker defaults stored in the checkpoint unless overridden.
    max_train = (args.max_train if args.max_train is not None
                 else train_cfg.get("dom_probe_max_train", 10000))
    max_test = (args.max_test if args.max_test is not None
                else train_cfg.get("dom_probe_max_test", 1500))
    sample_seed = (args.sample_seed if args.sample_seed is not None
                   else train_cfg.get("dom_probe_seed", 0))
    rng = torch.Generator(device="cpu").manual_seed(sample_seed)
    if max_train > 0 and max_train < train_idx.numel():
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=rng)[:max_train]]
    if max_test > 0 and max_test < test_idx.numel():
        test_idx = test_idx[torch.randperm(test_idx.numel(), generator=rng)[:max_test]]
    selected = torch.cat([train_idx, test_idx])
    n_train = train_idx.numel()
    edges = paths["edge_seqs"][selected]
    labels = paths["nodes"][selected, args.graph_layer].to(device)

    block = model.blocks[block_index]
    block_inputs, attention_outputs = [], []

    def save_block_input(_module, inputs):
        block_inputs.append(inputs[0].detach())

    def save_attention_output(_module, _inputs, output):
        attention_outputs.append(output.detach())

    pre_handle = block.register_forward_pre_hook(save_block_input)
    attn_handle = block.attn.register_forward_hook(save_attention_output)
    post_attention, post_block = [], []
    try:
        for start in range(0, edges.shape[0], args.batch_size):
            chunk = edges[start:start + args.batch_size].to(device)
            block_inputs.clear()
            attention_outputs.clear()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, hidden = model(chunk, return_hidden=True)
            a = block.drop(attention_outputs[0])
            after_attention = block_inputs[0] + a if block.use_residual else a
            post_attention.append(after_attention[:, args.position].float())
            post_block.append(hidden[args.block][:, args.position].float())
    finally:
        pre_handle.remove()
        attn_handle.remove()

    post_attention = torch.cat(post_attention)
    post_block = torch.cat(post_block)
    y_train, y_test = labels[:n_train], labels[n_train:]
    attn_acc, n_classes = _dom_top1(
        post_attention[:n_train], post_attention[n_train:], y_train, y_test)
    block_acc, _ = _dom_top1(
        post_block[:n_train], post_block[n_train:], y_train, y_test)

    print(f"checkpoint: {out_dir}")
    print(f"model: use_residual={model_cfg.use_residual}, use_mlp={model_cfg.use_mlp}")
    print(f"probe: graph_layer={args.graph_layer}, p={args.position}, "
          f"classes={n_classes}, train={n_train}, test={test_idx.numel()}")
    print(f"after attention {args.block}: DoM top-1 = {attn_acc:.3f}")
    print(f"after full block {args.block} (standard d={args.block}): "
          f"DoM top-1 = {block_acc:.3f}")


if __name__ == "__main__":
    main()
