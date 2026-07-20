"""Plot steering-intervention effectiveness over training for a target graph layer.

Requires a run that had --dom-steer-n-classes > 0. Reads dom_probe.pt and shows:
  - mean over K target classes of the steered-success rate over time
  - the clean (no-steering) baseline for the same rate
  - the uplift (steered - clean)
  - decodability (max top-1) on a twin axis for comparison
"""
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/dom_relu2_steer")
    ap.add_argument("--layer", type=int, default=-1,
                    help="Graph layer to plot; -1 -> last.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    p = torch.load(run_dir / "dom_probe.pt", weights_only=False)
    if "steer" not in p:
        raise SystemExit("no steering data in this run "
                         "(rerun with --dom-steer-n-classes > 0).")
    steps = p["step"]
    layers = p["target_layers"]
    ell = layers[-1] if args.layer < 0 else args.layer

    steer = p["steer"][ell]                  # (T, K)
    baseline = p["steer_baseline"][ell]      # (T, K)
    cell = p["steer_cell"][ell]              # (T, 2)  [d*, p*]
    top1 = p["top1"][ell]                    # (T, D, L)
    max_top1 = top1.view(top1.shape[0], -1).max(dim=1).values
    targets = p["steer_targets"][ell].tolist()

    steer_mean = steer.mean(dim=1)
    base_mean = baseline.mean(dim=1)
    uplift = steer_mean - base_mean
    alpha = p["cfg"]["steer_alpha"]

    print(f"layer {ell}: K={steer.shape[1]} target classes {targets}")
    print(f"final steered success (mean over K): {steer_mean[-1]:.3f}")
    print(f"final baseline           (mean over K): {base_mean[-1]:.3f}")
    print(f"final uplift                          : {uplift[-1]:.3f}")
    print(f"final steering cell (d*, p*)         : {cell[-1].tolist()}")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(steps.numpy(), steer_mean.numpy(), lw=2, color="C1",
            label=f"steered success  (alpha={alpha})")
    ax.plot(steps.numpy(), base_mean.numpy(), lw=1.6, ls="--", color="C7",
            label="clean baseline (no intervention)")
    ax.plot(steps.numpy(), uplift.numpy(), lw=1.4, color="C3",
            label="uplift (steered - clean)")

    ax2 = ax.twinx()
    ax2.plot(steps.numpy(), max_top1.numpy(), lw=1.2, color="C0", alpha=0.7,
             label="max DoM top-1 (decodability)")
    ax2.set_ylabel("DoM top-1 accuracy", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel("training step")
    ax.set_ylabel("P(prediction reachable from target class)")
    ax.set_title(f"Layer {ell} steering intervention "
                 f"(K={steer.shape[1]}, alpha={alpha}, "
                 f"intervene at each layer's best DoM cell)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)

    out = Path(args.out) if args.out else run_dir / f"dom_steer_layer{ell}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
