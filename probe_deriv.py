"""Plot DoM multi-class top-1 accuracy over training for representation,
difference, and derivative, at a target graph layer.

Reads out_dir/dom_probe.pt and writes out_dir/probe_deriv.png.

Matches plot_dom_layer.py's aggregation for the rep line:
    at each probe step, max over transformer (depth, position) of the (D, L)
    top-1 cube; then plot vs step.

    rep(t)   = h(t)
    diff(t)  = h(t) - h(prev probe step)
    deriv(t) = h(t) - h(t - 1 training step)
    deriv10(t) = h(t) - h(t - 10 training steps)
    deriv50(t) = h(t) - h(t - 50 training steps)
"""
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cell_line(tensor: torch.Tensor, d_star: int, p_star: int) -> torch.Tensor:
    """tensor: (T, D, L). Returns (T,) at fixed (d_star, p_star)."""
    return tensor[:, d_star, p_star]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 4],
                    help="Graph layers to plot (default: 3 4).")
    ap.add_argument("--rep-compare-layers", type=int, nargs="+",
                    default=[2, 3, 4, 5],
                    help="Layers for the rep-only cross-layer comparison plot.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    payload = torch.load(out_dir / "dom_probe.pt", weights_only=False)
    steps = payload["step"]
    layers = payload["target_layers"]

    for ell in args.layers:
        if ell not in layers:
            print(f"layer {ell} not tracked (available: {layers}); skipping")
            continue
        _render_layer(ell, payload, steps, out_dir)
        _render_loss_grad_vs_rep(ell, payload, steps, out_dir)

    cmp_layers = [ell for ell in args.rep_compare_layers if ell in layers]
    if cmp_layers:
        _render_rep_across_layers(cmp_layers, payload, steps, out_dir)
        _render_final_probe_across_layers(cmp_layers, payload, steps, out_dir)
        _render_final_probe_across_model_depth(cmp_layers, payload, out_dir)


def _render_layer(ell, payload, steps, out_dir):
    n_classes = payload["reachable_classes"][ell].numel()
    chance = 1.0 / n_classes

    final_top1 = payload["final_top1_rep"][ell]         # (D, L)
    L = final_top1.shape[1]
    best_idx = final_top1.reshape(-1).argmax().item()
    d_star, p_star = best_idx // L, best_idx % L
    print(f"layer {ell}: best cell (d, p) = ({d_star}, {p_star})  "
          f"final DoM top-1 = {final_top1[d_star, p_star].item():.3f}")

    feature_defs = [
        (r"rep $h(t)$", "rep", "step", "C0"),
        (r"diff $h(t)-h(t_\mathrm{prev\ probe})$", "diff", "step", "C1"),
        (r"deriv $h(t)-h(t{-}1)$", "deriv", "deriv_step", "C2"),
        (r"deriv10 $h(t)-h(t{-}10)$", "deriv10", "deriv10_step", "C3"),
        (r"deriv50 $h(t)-h(t{-}50)$", "deriv50", "deriv50_step", "C4"),
        (r"$-\partial L/\partial h$", "loss_grad", "step", "C5"),
    ]

    steps_np = steps.numpy()
    dom_curves = payload.get("final_dom_top1_curve", {})
    lr_curves = payload.get("final_lr_top1_curve", {})
    fig, ax = plt.subplots(figsize=(9, 5))
    have_any = False
    have_any_lr = False
    for name, key, step_key, color in feature_defs:
        dom_curve = dom_curves.get(key, {}).get(ell)
        lr_curve = lr_curves.get(key, {}).get(ell)
        if dom_curve is None and lr_curve is None:
            continue
        t_line = (payload[step_key].numpy()
                  if step_key in payload else steps_np)
        if dom_curve is not None and dom_curve.numel() > 0:
            ax.plot(t_line[:dom_curve.shape[0]], dom_curve.numpy(),
                    color=color, lw=2, label=f"DoM {name}")
            have_any = True
        if lr_curve is not None and lr_curve.numel() > 0:
            ax.plot(t_line[:lr_curve.shape[0]], lr_curve.numpy(),
                    color=color, lw=1.6, ls="--", label=f"logreg {name}")
            have_any_lr = True
    ax.axhline(chance, color="grey", ls=":", lw=1,
               label=f"chance ({chance:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"top-1 accuracy  (cell d={d_star}, p={p_star})")
    if not have_any:
        subtitle = "final-applied curves unavailable"
    elif have_any_lr:
        subtitle = "final-applied: solid DoM / dashed logreg  (both fit at end)"
    else:
        subtitle = "final-applied DoM only"
    ax.set_title(f"Layer {ell} decodability  ({n_classes} classes)  "
                 f"at cell (d={d_star}, p={p_star})  — {subtitle}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plot_path = out_dir / f"probe_deriv_{ell}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved plot -> {plot_path}  (layer {ell})")


def _render_loss_grad_vs_rep(ell, payload, steps, out_dir):
    """Compare probes refit at each checkpoint on h and on -dL/dh."""
    rep = payload.get("top1", {}).get(ell)
    loss_grad = payload.get("top1_loss_grad", {}).get(ell)
    if rep is None or loss_grad is None or rep.numel() == 0 or loss_grad.numel() == 0:
        print(f"layer {ell}: contemporaneous rep/loss-gradient probes unavailable; skipping")
        return

    count = min(rep.shape[0], loss_grad.shape[0], steps.numel())
    rep = rep[:count]
    loss_grad = loss_grad[:count]
    x = steps[:count].numpy()

    # Fix the comparison cell using the final h(t) probe only, so selection does
    # not favor the loss-gradient probe and both curves use the identical cell.
    final_rep = rep[-1]
    width = final_rep.shape[1]
    best_idx = final_rep.reshape(-1).argmax().item()
    d_star, p_star = best_idx // width, best_idx % width

    rep_best = rep.flatten(1).max(dim=1).values.numpy()
    grad_best = loss_grad.flatten(1).max(dim=1).values.numpy()
    rep_fixed = rep[:, d_star, p_star].numpy()
    grad_fixed = loss_grad[:, d_star, p_star].numpy()
    chance = 1.0 / payload["reachable_classes"][ell].numel()

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, rep_best, lw=2.2, color="C0", label=r"DoM on $h(t)$")
    axes[0].plot(x, grad_best, lw=2.2, color="C5",
                 label=r"DoM on $-\partial L/\partial h$")
    axes[0].set_title("Best model-layer/position cell at each checkpoint")

    axes[1].plot(x, rep_fixed, lw=2.2, color="C0", label=r"DoM on $h(t)$")
    axes[1].plot(x, grad_fixed, lw=2.2, color="C5",
                 label=r"DoM on $-\partial L/\partial h$")
    axes[1].set_title(f"Same fixed cell: model layer {d_star}, position {p_star}")
    axes[1].set_xlabel("training step")

    for ax in axes:
        ax.axhline(chance, color="grey", ls=":", lw=1,
                   label=f"chance ({chance:.3f})")
        ax.set_ylabel("held-out top-1 accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(
        f"Graph layer {ell}: contemporaneous DoM probe on representation vs loss descent"
    )
    plot_path = out_dir / f"probe_dom_loss_grad_vs_rep_{ell}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"saved plot -> {plot_path}  (layer {ell}, probes refit every checkpoint)")


def _render_rep_across_layers(layers, payload, steps, out_dir):
    """One panel: rep h(t) decodability at each layer's best cell, DoM+logreg."""
    steps_np = steps.numpy()
    dom_curves = payload.get("final_dom_top1_curve", {}).get("rep", {})
    lr_curves = payload.get("final_lr_top1_curve", {}).get("rep", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    n = max(len(layers) - 1, 1)
    have_any_lr = False
    chances = []
    for i, ell in enumerate(layers):
        color = cmap(i / n)
        final_top1 = payload["final_top1_rep"][ell]
        L = final_top1.shape[1]
        best_idx = final_top1.reshape(-1).argmax().item()
        d_star, p_star = best_idx // L, best_idx % L
        n_classes = payload["reachable_classes"][ell].numel()
        chances.append((ell, 1.0 / n_classes, color))
        suffix = f"L{ell} (d={d_star}, p={p_star}, {n_classes} cls)"

        dom_curve = dom_curves.get(ell)
        lr_curve = lr_curves.get(ell)
        if dom_curve is not None and dom_curve.numel() > 0:
            ax.plot(steps_np[:dom_curve.shape[0]], dom_curve.numpy(),
                    color=color, lw=2, label=f"DoM {suffix}")
        if lr_curve is not None and lr_curve.numel() > 0:
            ax.plot(steps_np[:lr_curve.shape[0]], lr_curve.numpy(),
                    color=color, lw=1.6, ls="--", label=f"logreg {suffix}")
            have_any_lr = True

    for ell, ch, color in chances:
        ax.axhline(ch, color=color, ls=":", lw=1, alpha=0.6)

    ax.set_xlabel("training step")
    ax.set_ylabel("top-1 accuracy (rep, best cell per layer)")
    subtitle = ("final-applied: solid DoM / dashed logreg"
                if have_any_lr else "final-applied DoM only")
    ax.set_title(f"Rep $h(t)$ decodability across layers {layers}  — {subtitle}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plot_path = out_dir / "probe_deriv_rep_layers.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved plot -> {plot_path}  (layers {layers})")


def _render_final_probe_across_layers(layers, payload, steps, out_dir):
    """One panel: final-applied DoM representation probe for each graph layer."""
    steps_np = steps.numpy()
    dom_curves = payload.get("final_dom_top1_curve", {}).get("rep", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    n = max(len(layers) - 1, 1)
    for i, ell in enumerate(layers):
        curve = dom_curves.get(ell)
        if curve is None or curve.numel() == 0:
            continue
        color = cmap(i / n)
        final_top1 = payload["final_top1_rep"][ell]
        width = final_top1.shape[1]
        best_idx = final_top1.reshape(-1).argmax().item()
        d_star, p_star = best_idx // width, best_idx % width
        ax.plot(
            steps_np[:curve.shape[0]], curve.numpy(),
            color=color, lw=2.2,
            label=f"Layer {ell} (d={d_star}, p={p_star}) — {curve[-1].item():.3f}",
        )

    ax.set_xlabel("training step")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Final representation probe performance across graph layers")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path = out_dir / "probe_final_rep_layers_2_to_5.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"saved plot -> {plot_path}  (layers {layers}, final DoM probe only)")


def _render_final_probe_across_model_depth(layers, payload, out_dir):
    """Final-checkpoint probe accuracy vs transformer depth for graph layers."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    n = max(len(layers) - 1, 1)

    for i, ell in enumerate(layers):
        # (model depth, sequence position); select the best position separately
        # at each depth so the curve measures the strongest representation there.
        depth_position = payload["final_top1_rep"][ell]
        depth_curve = depth_position.max(dim=1).values
        model_layers = torch.arange(depth_curve.numel())
        ax.plot(
            model_layers.numpy(), depth_curve.numpy(),
            color=cmap(i / n), marker="o", lw=2.2,
            label=f"Graph layer {ell}",
        )

    ax.set_xlabel("model layer (transformer depth)")
    ax.set_ylabel("final-probe top-1 accuracy")
    ax.set_xticks(range(payload["final_top1_rep"][layers[0]].shape[0]))
    ax.set_ylim(0, 1)
    ax.set_title("Final representation probe across model layers")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path = out_dir / "probe_final_graph_layers_2_to_5_across_model_layers.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"saved plot -> {plot_path}  (x-axis=model layer, graph layers={layers})")


if __name__ == "__main__":
    main()
