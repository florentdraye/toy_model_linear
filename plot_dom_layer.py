"""Plot max-over-(depth, position) DoM multi-class accuracy vs training step
for a target graph layer.

Reads runs/<name>/dom_probe.pt (which now includes top1 and top5 fields per
target layer) and produces a PNG showing when the target latents first become
linearly decodable and how the best-cell probe evolves.
"""
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def find_first_cross(vals: torch.Tensor, steps: torch.Tensor, thr: float):
    hit = vals >= thr
    if not hit.any():
        return None
    return int(steps[hit.nonzero()[0].item()].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/dom_analysis")
    ap.add_argument("--layer", type=int, default=-1,
                    help="Graph layer to plot; -1 -> last.")
    ap.add_argument("--out", type=str, default=None,
                    help="PNG output path (default: <run>/dom_layer<L>.png).")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="Threshold for reporting first-crossing step.")
    ap.add_argument("--max-step", type=int, default=0,
                    help="Cap x-axis at this training step (0 = no cap).")
    args = ap.parse_args()

    run_dir = Path(args.run)
    payload = torch.load(run_dir / "dom_probe.pt", weights_only=False)
    # Optional: load model train/test loss log from ckpt.pt for a side panel.
    log = None
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        log = ckpt.get("log")
    steps = payload["step"]
    layers = payload["target_layers"]
    ell = layers[-1] if args.layer < 0 else args.layer
    assert ell in layers, f"layer {ell} not tracked; got {layers}"

    top1 = payload["top1"][ell]                      # (T, D, L)
    top5 = payload["top5"][ell]
    T, D, L = top1.shape
    n_classes = payload["reachable_classes"][ell].numel()
    chance1, chance5 = 1.0 / n_classes, 5.0 / n_classes

    # Max and mean over (d, p) at each step.
    max_top1 = top1.view(T, -1).max(dim=1).values
    max_top5 = top5.view(T, -1).max(dim=1).values
    mean_top1 = top1.view(T, -1).mean(dim=1)
    mean_top5 = top5.view(T, -1).mean(dim=1)
    # Which (d, p) is best at the end?
    final_flat = top1[-1].view(-1)
    best_idx = final_flat.argmax().item()
    bd, bp = best_idx // L, best_idx % L

    first_cross = find_first_cross(max_top1, steps, args.thr)
    print(f"layer {ell}: {n_classes} classes  (chance top-1={chance1:.3f}, "
          f"top-5={chance5:.3f})")
    print(f"final best cell: (d={bd}, p={bp})  top-1={top1[-1, bd, bp]:.3f}")
    print(f"first max-top-1 >= {args.thr}: "
          f"{'never' if first_cross is None else f'step {first_cross}'}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    ax = axes[0]
    ax.plot(steps.numpy(), max_top1.numpy(), lw=2,
            label=f"DoM max over (d, p)  [best cell (d={bd},p={bp})]")
    if "logreg_top1" in payload:
        lr_max = payload["logreg_top1"][ell].view(T, -1).max(dim=1).values
        ax.plot(steps.numpy(), lr_max.numpy(), lw=2, color="C3",
                label="LogReg max over (d, p)")
    if "mlp_top1" in payload:
        mlp_max = payload["mlp_top1"][ell].view(T, -1).max(dim=1).values
        ax.plot(steps.numpy(), mlp_max.numpy(), lw=2, color="C2",
                label="MLP max over (d, p)")
    ax.axhline(chance1, color="grey", ls="--", lw=1, label=f"chance ({chance1:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("DoM top-1 accuracy")
    ax.set_title(f"Layer {ell} decodability (top-1, {n_classes} classes)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if log is not None:
        log_steps = log["step"]
        ax.plot(log_steps, log["train_loss"], lw=2, color="C0",
                label="train loss")
        ax.plot(log_steps, log["test_loss"], lw=2, color="C3",
                label="test loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Model train/test loss")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    if args.max_step > 0:
        for a in axes:
            a.set_xlim(0, args.max_step)

    out = Path(args.out) if args.out else run_dir / f"dom_layer{ell}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
