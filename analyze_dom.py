"""Quick analysis of a dom_probe.pt log produced by train.py --dom-probe-every-steps.

Prints for each target graph layer:
  - the initial vs final mean AUC per (depth, position)
  - the step at which the mean AUC first crosses a threshold (e.g. 0.9)
  - the top-K (depth, position) cells and their AUC trajectory summary.
"""
import argparse
from pathlib import Path

import torch


def nanmean(x, dim=None):
    m = torch.isnan(x)
    return torch.where(m, torch.zeros_like(x), x).sum(dim) / (~m).float().sum(dim).clamp(min=1)


def fmt_grid(mat: torch.Tensor, row_label: str = "d") -> str:
    D, L = mat.shape
    header = f"  {row_label}\\p  " + " ".join(f" p{p}" for p in range(L))
    lines = [header, "  " + "-" * (len(header) - 2)]
    for d in range(D):
        row = " ".join(f"{mat[d, p].item():5.3f}" if not torch.isnan(mat[d, p])
                       else "  nan" for p in range(L))
        lines.append(f"  d={d}   {row}")
    return "\n".join(lines)


def first_cross_step(traj: torch.Tensor, steps: torch.Tensor, thr: float) -> int:
    """Return the earliest step at which trajectory crosses `thr`, else -1."""
    hit = traj >= thr
    if not hit.any():
        return -1
    return int(steps[hit.nonzero()[0].item()].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/dom_analysis")
    ap.add_argument("--thr", type=float, default=0.9,
                    help="AUC threshold for reporting first-crossing step.")
    ap.add_argument("--topk", type=int, default=5,
                    help="Top-K (depth, position) cells to report per layer.")
    args = ap.parse_args()

    path = Path(args.run) / "dom_probe.pt"
    p = torch.load(path, weights_only=False)
    steps = p["step"]                              # (T,)
    T = steps.numel()

    print(f"loaded {path}")
    print(f"  probe steps: {T}   range [{steps[0].item()}, {steps[-1].item()}]")
    print(f"  target graph layers: {p['target_layers']}")
    print(f"  cfg: {p['cfg']}\n")

    for ell in p["target_layers"]:
        cube = p["cubes"][ell]                     # (T, D, L, K)
        T_, D, L, K = cube.shape
        assert T_ == T
        chosen = p["chosen"][ell]
        # Mean over sampled nodes -> (T, D, L)
        cube_mean = nanmean(cube, dim=-1)
        init = cube_mean[0]                        # (D, L)
        final = cube_mean[-1]
        peak = cube_mean.max(dim=0).values         # (D, L)  best over time
        peak_step_idx = cube_mean.argmax(dim=0)    # (D, L)  step-idx at peak

        print(f"=== graph layer {ell}   ({K} sampled nodes, ids={chosen.tolist()}) ===")
        print("initial mean AUC:")
        print(fmt_grid(init))
        print("\nfinal   mean AUC:")
        print(fmt_grid(final))
        print("\npeak    mean AUC (across training):")
        print(fmt_grid(peak))

        # Per-cell trajectories: rank cells by peak AUC, print top-K.
        flat_peak = peak.flatten()
        best_idx = flat_peak.argsort(descending=True)[:args.topk]
        print(f"\ntop-{args.topk} (depth,pos) cells by peak AUC:")
        for i in best_idx.tolist():
            d, ppos = i // L, i % L
            traj = cube_mean[:, d, ppos]
            fs = first_cross_step(traj, steps, args.thr)
            fs_str = "never" if fs < 0 else f"step {fs}"
            print(f"  d={d} p={ppos}: init={traj[0].item():.3f}  peak={traj.max().item():.3f}  "
                  f"final={traj[-1].item():.3f}  first>={args.thr}: {fs_str}")

        # Per-node AUC at end: identify latents that got separated vs not.
        final_per_node = nanmean(cube[-1], dim=(0, 1))  # (K,)  mean over (d, p)
        peak_per_node = nanmean(cube.max(dim=0).values, dim=(0, 1))
        order = peak_per_node.argsort(descending=True)
        print(f"\nper-node final vs peak AUC (mean over d,p), sorted by peak:")
        for k in order.tolist():
            print(f"  node {chosen[k].item():4d}: final={final_per_node[k].item():.3f}  "
                  f"peak={peak_per_node[k].item():.3f}")
        print()


if __name__ == "__main__":
    main()
