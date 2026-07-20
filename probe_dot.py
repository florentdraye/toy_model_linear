"""Plot cosine(w_c^final, X(t)) over training for X in {h(t), h(t)-h(prev probe),
h(t)-h(t-1 step), h(t)-h(t-10 steps), h(t)-h(t-50 steps), -dL/dh},
split by active vs inactive samples.
Also plots raw <w_c^final, X(t)> dot products in a bottom row.

Reads out_dir/dom_probe.pt (must include dot_{rep,diff,deriv}_{active,inactive})
and writes out_dir/probe_dot.png.

Top row: cosine alignment. Bottom row: raw dot product.
    active(t)   = mean_c   mean_{x: y_ell(x) = c}  cos(w_c^final, X_t(x))
    inactive(t) = mean_c   mean_{x: y_ell(x) != c} cos(w_c^final, X_t(x))
The shaded bands are std across classes. The (d, p) cell is fixed to the
argmax of the final multi-class top-1 accuracy over (d, p) at layer ell.
"""
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mean_std_at_cell(active: torch.Tensor, inactive: torch.Tensor,
                      d_star: int, p_star: int
                      ) -> tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor]:
    """active, inactive: (T, D, L, K). Slice at (d*, p*), then reduce over K."""
    a = active[:, d_star, p_star, :]          # (T, K)
    i = inactive[:, d_star, p_star, :]
    return a.mean(dim=1), a.std(dim=1), i.mean(dim=1), i.std(dim=1)


def _draw_panel(ax, steps_np, mean_a, std_a, mean_i, std_i, title):
    ax.plot(steps_np[:mean_a.shape[0]], mean_a.numpy(), color="C3", lw=2,
            label=r"active: $y_\ell(x) = c$")
    ax.fill_between(steps_np[:mean_a.shape[0]],
                    (mean_a - std_a).numpy(), (mean_a + std_a).numpy(),
                    color="C3", alpha=0.2)
    ax.plot(steps_np[:mean_i.shape[0]], mean_i.numpy(), color="C0", lw=2,
            label=r"inactive: $y_\ell(x) \neq c$")
    ax.fill_between(steps_np[:mean_i.shape[0]],
                    (mean_i - std_i).numpy(), (mean_i + std_i).numpy(),
                    color="C0", alpha=0.2)
    ax.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _draw_fraction_panel(ax, steps_np, curves, title):
    colors = ["C2", "C4", "C1", "C5", "C6", "C7"]
    for idx, (frac, mean_a, mean_i) in enumerate(curves):
        color = colors[idx % len(colors)]
        label = f"{100.0 * frac:g}% active"
        ax.plot(steps_np[:mean_a.shape[0]], mean_a.numpy(), color=color,
                lw=2, label=label)
        ax.plot(steps_np[:mean_i.shape[0]], mean_i.numpy(), color=color,
                lw=1.4, ls="--", alpha=0.75)
    ax.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _probe_payload_is_single_fraction(probe_active: dict, ell: int) -> bool:
    return ell in probe_active and isinstance(probe_active[ell], torch.Tensor)


def _cell_stats_or_none(payload, active_key, inactive_key, ell, d_star, p_star):
    if active_key not in payload:
        return None
    active = payload[active_key][ell]
    inactive = payload[inactive_key][ell]
    if active.numel() == 0:
        return None
    return _mean_std_at_cell(active, inactive, d_star, p_star)


def _mean_std_from_tK(active_tK: torch.Tensor, inactive_tK: torch.Tensor):
    """active/inactive: (T, K). Returns (mean_a, std_a, mean_i, std_i)."""
    return (active_tK.mean(dim=1), active_tK.std(dim=1),
            inactive_tK.mean(dim=1), inactive_tK.std(dim=1))


def _lr_stats_or_none(payload, active_key, inactive_key, ell):
    if active_key not in payload or ell not in payload[active_key]:
        return None
    active = payload[active_key][ell]
    inactive = payload[inactive_key][ell]
    if active.numel() == 0:
        return None
    return _mean_std_from_tK(active, inactive)


def _lr_ntk_pair_curves(payload, key, ell):
    """LR-based NTK pair curves. Cubes are (T, K); reduce over K to a mean."""
    if key not in payload:
        return None
    labels = {
        "ta_sa": "target active / source active",
        "ta_si": "target active / source inactive",
        "ti_sa": "target inactive / source active",
        "ti_si": "target inactive / source inactive",
    }
    curves = []
    for combo, label in labels.items():
        if combo not in payload[key] or ell not in payload[key][combo]:
            return None
        cube = payload[key][combo][ell]
        if cube.numel() == 0:
            return None
        curves.append((label, cube.mean(dim=1)))
    return curves


def _ntk_pair_curves(payload, key, ell, d_star, p_star):
    if key not in payload:
        return None
    labels = {
        "ta_sa": "target active / source active",
        "ta_si": "target active / source inactive",
        "ti_sa": "target inactive / source active",
        "ti_si": "target inactive / source inactive",
    }
    curves = []
    for combo, label in labels.items():
        if combo not in payload[key] or ell not in payload[key][combo]:
            return None
        cube = payload[key][combo][ell]
        if cube.numel() == 0:
            return None
        vals = cube[:, d_star, p_star, :].mean(dim=1)
        curves.append((label, vals))
    return curves


def _draw_ntk_pair_panel(ax, steps_np, curves, title):
    colors = ["C3", "C1", "C0", "C4"]
    for (label, vals), color in zip(curves, colors):
        ax.plot(steps_np[:vals.shape[0]], vals.numpy(), color=color, lw=2,
                label=label)
    ax.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 4],
                    help="Graph layers to analyse (default: 3 4).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    payload = torch.load(out_dir / "dom_probe.pt", weights_only=False)
    train_log = None
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        train_log = ckpt.get("log")
    if "dot_rep_active" not in payload:
        raise RuntimeError(
            "dom_probe.pt has no dot_rep_active field. Re-run training with "
            "the current tracker."
        )
    steps = payload["step"]
    layers = payload["target_layers"]
    steps_np = steps.numpy()

    for ell in args.layers:
        if ell not in layers:
            print(f"layer {ell} not tracked (available: {layers}); skipping")
            continue
        _render_layer(ell, payload, steps, steps_np, train_log, out_dir)


def _render_layer(ell, payload, steps, steps_np, train_log, out_dir):
    final_top1 = payload["final_top1_rep"][ell]      # (D, L)
    T_rep, D, L, K = payload["dot_rep_active"][ell].shape
    best_idx = final_top1.reshape(-1).argmax().item()
    d_star, p_star = best_idx // L, best_idx % L
    print(f"layer {ell}: best cell (d, p) = ({d_star}, {p_star})  "
          f"final top-1 = {final_top1[d_star, p_star].item():.3f}")

    rep_mA, rep_sA, rep_mI, rep_sI = _mean_std_at_cell(
        payload["dot_rep_active"][ell],
        payload["dot_rep_inactive"][ell], d_star, p_star)
    diff_mA, diff_sA, diff_mI, diff_sI = _mean_std_at_cell(
        payload["dot_diff_active"][ell],
        payload["dot_diff_inactive"][ell], d_star, p_star)
    deriv_a_cube = payload["dot_deriv_active"][ell]
    deriv_i_cube = payload["dot_deriv_inactive"][ell]
    deriv_mA, deriv_sA, deriv_mI, deriv_sI = _mean_std_at_cell(
        deriv_a_cube, deriv_i_cube, d_star, p_star)
    deriv_steps = payload.get("deriv_step", steps[:deriv_mA.shape[0]]).numpy()
    deriv10 = None
    if "dot_deriv10_active" in payload:
        deriv10_a_cube = payload["dot_deriv10_active"][ell]
        deriv10_i_cube = payload["dot_deriv10_inactive"][ell]
        if deriv10_a_cube.numel() > 0:
            deriv10 = _mean_std_at_cell(
                deriv10_a_cube, deriv10_i_cube, d_star, p_star)
            deriv10_steps = payload.get(
                "deriv10_step", steps[:deriv10[0].shape[0]]
            ).numpy()
    deriv50 = None
    if "dot_deriv50_active" in payload:
        deriv50_a_cube = payload["dot_deriv50_active"][ell]
        deriv50_i_cube = payload["dot_deriv50_inactive"][ell]
        if deriv50_a_cube.numel() > 0:
            deriv50 = _mean_std_at_cell(
                deriv50_a_cube, deriv50_i_cube, d_star, p_star)
            deriv50_steps = payload.get(
                "deriv50_step", steps[:deriv50[0].shape[0]]
            ).numpy()
    loss_grad = _cell_stats_or_none(
        payload, "dot_loss_grad_active", "dot_loss_grad_inactive",
        ell, d_star, p_star)
    ntk_cos = _ntk_pair_curves(payload, "dot_ntk_pair", ell, d_star, p_star)
    ntk_raw = _ntk_pair_curves(payload, "raw_dot_ntk_pair", ell, d_star, p_star)
    ntk_grad_cos = _ntk_pair_curves(
        payload, "dot_ntk_grad_pair", ell, d_star, p_star)
    ntk_grad_raw = _ntk_pair_curves(
        payload, "raw_dot_ntk_grad_pair", ell, d_star, p_star)

    raw_panels = [
        _cell_stats_or_none(payload, "raw_dot_rep_active", "raw_dot_rep_inactive",
                            ell, d_star, p_star),
        _cell_stats_or_none(payload, "raw_dot_diff_active", "raw_dot_diff_inactive",
                            ell, d_star, p_star),
        _cell_stats_or_none(payload, "raw_dot_deriv_active", "raw_dot_deriv_inactive",
                            ell, d_star, p_star),
        _cell_stats_or_none(payload, "raw_dot_deriv10_active", "raw_dot_deriv10_inactive",
                            ell, d_star, p_star),
        _cell_stats_or_none(payload, "raw_dot_deriv50_active", "raw_dot_deriv50_inactive",
                            ell, d_star, p_star),
        _cell_stats_or_none(payload, "raw_dot_loss_grad_active", "raw_dot_loss_grad_inactive",
                            ell, d_star, p_star),
    ]

    fig, axes = plt.subplots(4, 9, figsize=(39.5, 16.4), sharex=False)
    _draw_panel(axes[0, 0], steps_np, rep_mA, rep_sA, rep_mI, rep_sI,
                r"representation $h(t)$")
    _draw_panel(axes[0, 1], steps_np, diff_mA, diff_sA, diff_mI, diff_sI,
                r"$\Delta$ checkpoint  $h(t) - h(t_\mathrm{prev\ probe})$")
    _draw_panel(axes[0, 2], deriv_steps, deriv_mA, deriv_sA, deriv_mI, deriv_sI,
                r"$\Delta_t$ actual update  $h(t) - h(t{-}1)$")
    if deriv10 is not None:
        deriv10_mA, deriv10_sA, deriv10_mI, deriv10_sI = deriv10
        _draw_panel(axes[0, 3], deriv10_steps, deriv10_mA, deriv10_sA,
                    deriv10_mI, deriv10_sI,
                    r"$\Delta_{10}$ actual window  $h(t) - h(t{-}10)$")
    else:
        axes[0, 3].set_axis_off()
        axes[0, 3].set_title(r"$\Delta_{10}$ unavailable")
    if deriv50 is not None:
        deriv50_mA, deriv50_sA, deriv50_mI, deriv50_sI = deriv50
        _draw_panel(axes[0, 4], deriv50_steps, deriv50_mA, deriv50_sA,
                    deriv50_mI, deriv50_sI,
                    r"$\Delta_{50}$ actual window  $h(t) - h(t{-}50)$")
    else:
        axes[0, 4].set_axis_off()
        axes[0, 4].set_title(r"$\Delta_{50}$ unavailable")
    if loss_grad is not None:
        loss_mA, loss_sA, loss_mI, loss_sI = loss_grad
        _draw_panel(axes[0, 5], steps_np, loss_mA, loss_sA, loss_mI, loss_sI,
                    r"activation descent  $-\partial L/\partial h$")
    else:
        axes[0, 5].set_axis_off()
        axes[0, 5].set_title(r"$-\partial L/\partial h$ unavailable")
    if ntk_grad_cos is not None:
        _draw_ntk_pair_panel(
            axes[0, 6], steps_np, ntk_grad_cos,
            r"pair NTK loss-gradient  $K_{ij}(-\partial L_j/\partial h_j)$"
        )
    else:
        axes[0, 6].set_axis_off()
        axes[0, 6].set_title("loss-gradient pair NTK unavailable")
    if ntk_cos is not None:
        _draw_ntk_pair_panel(
            axes[0, 7], steps_np, ntk_cos,
            r"pair NTK eigentest  $K_{ij}w_c^{final}$"
        )
    else:
        axes[0, 7].set_axis_off()
        axes[0, 7].set_title("pair NTK eigentest unavailable")
    axes[0, 0].set_ylabel(r"$\cos(w_c^{\mathrm{final}},\;X(t))$"
                          "  (mean $\\pm$ std over K classes)")

    dot_steps = [steps_np, steps_np, deriv_steps,
                 locals().get("deriv10_steps"), locals().get("deriv50_steps"),
                 steps_np]
    dot_titles = [
        r"raw dot  $h(t)$",
        r"raw dot  $h(t) - h(t_\mathrm{prev\ probe})$",
        r"raw dot  $h(t) - h(t{-}1)$",
        r"raw dot  $h(t) - h(t{-}10)$",
        r"raw dot  $h(t) - h(t{-}50)$",
        r"raw dot  $-\partial L/\partial h$",
    ]
    for j, stats in enumerate(raw_panels):
        if stats is None or dot_steps[j] is None:
            axes[1, j].set_axis_off()
            axes[1, j].set_title("raw dot unavailable")
            continue
        mA, sA, mI, sI = stats
        _draw_panel(axes[1, j], dot_steps[j], mA, sA, mI, sI, dot_titles[j])
    if ntk_grad_raw is not None:
        _draw_ntk_pair_panel(
            axes[1, 6], steps_np, ntk_grad_raw,
            r"raw dot loss-gradient pair NTK"
        )
    else:
        axes[1, 6].set_axis_off()
        axes[1, 6].set_title("raw dot loss-gradient pair NTK unavailable")
    if ntk_raw is not None:
        _draw_ntk_pair_panel(
            axes[1, 7], steps_np, ntk_raw,
            r"raw dot pair NTK eigentest"
        )
    else:
        axes[1, 7].set_axis_off()
        axes[1, 7].set_title("raw dot pair NTK eigentest unavailable")
    axes[1, 0].set_ylabel(r"$\langle w_c^{\mathrm{final}},\;X(t)\rangle$"
                          "  (mean $\\pm$ std over K classes)")
    if ntk_cos is not None:
        axes[0, 7].legend(loc="best", fontsize=9)
    elif ntk_grad_cos is not None:
        axes[0, 6].legend(loc="best", fontsize=9)
    elif loss_grad is not None:
        axes[0, 5].legend(loc="best", fontsize=9)
    elif deriv50 is not None:
        axes[0, 4].legend(loc="best", fontsize=9)
    elif deriv10 is not None:
        axes[0, 3].legend(loc="best", fontsize=9)
    else:
        axes[0, 2].legend(loc="best", fontsize=9)

    if train_log is not None and len(train_log.get("step", [])) > 0:
        log_steps = train_log["step"]
        axes[0, 8].plot(log_steps, train_log["train_loss"], color="C3",
                        lw=2, label="train")
        axes[0, 8].plot(log_steps, train_log["test_loss"], color="C0",
                        lw=2, label="test")
        axes[0, 8].set_xlabel("training step")
        axes[0, 8].set_title("cross-entropy loss")
        axes[0, 8].grid(True, alpha=0.3)
        axes[0, 8].legend(loc="best", fontsize=9)
        axes[1, 8].plot(log_steps, train_log["train_acc"], color="C3",
                        lw=2, label="train")
        axes[1, 8].plot(log_steps, train_log["test_acc"], color="C0",
                        lw=2, label="test")
        axes[1, 8].set_xlabel("training step")
        axes[1, 8].set_title("top-1 accuracy")
        axes[1, 8].set_ylim(0.0, 1.0)
        axes[1, 8].grid(True, alpha=0.3)
    else:
        axes[0, 8].set_axis_off()
        axes[0, 8].set_title("loss log unavailable")
        axes[1, 8].set_axis_off()
        axes[1, 8].set_title("accuracy log unavailable")

    _draw_lr_rows(axes, payload, ell, steps_np, deriv_steps,
                  locals().get("deriv10_steps"),
                  locals().get("deriv50_steps"))

    fig.suptitle(f"Layer {ell}, cell (d={d_star}, p={p_star})  —  "
                 f"rows 1-2: DoM $w_c^{{final}}$;  "
                 f"rows 3-4: multinomial-softmax logreg $w_c^{{final}}$  "
                 f"(K={K} probe classes)")

    plot_path = out_dir / f"probe_dot_{ell}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved plot -> {plot_path}  "
          f"(layer {ell}, cell d={d_star}, p={p_star})")

    _plot_per_class_loss_grad(payload, ell, d_star, p_star, steps_np, out_dir)


def _draw_lr_rows(axes, payload, ell, steps_np, deriv_steps,
                  deriv10_steps, deriv50_steps):
    """Fill rows 2 (cosine) and 3 (raw dot) with logistic-regression w_c
    versions of the same panels as rows 0 and 1. LR cubes are stored at the
    DoM-selected cell already, shape (T, K)."""
    if "dot_lr_rep_active" not in payload:
        for r in (2, 3):
            for j in range(9):
                axes[r, j].set_axis_off()
                axes[r, j].set_title("logreg direction unavailable")
        return

    rep = _lr_stats_or_none(payload, "dot_lr_rep_active",
                            "dot_lr_rep_inactive", ell)
    diff = _lr_stats_or_none(payload, "dot_lr_diff_active",
                             "dot_lr_diff_inactive", ell)
    deriv = _lr_stats_or_none(payload, "dot_lr_deriv_active",
                              "dot_lr_deriv_inactive", ell)
    deriv10 = _lr_stats_or_none(payload, "dot_lr_deriv10_active",
                                "dot_lr_deriv10_inactive", ell)
    deriv50 = _lr_stats_or_none(payload, "dot_lr_deriv50_active",
                                "dot_lr_deriv50_inactive", ell)
    lg = _lr_stats_or_none(payload, "dot_lr_loss_grad_active",
                           "dot_lr_loss_grad_inactive", ell)
    ntk_grad = _lr_ntk_pair_curves(payload, "dot_lr_ntk_grad_pair", ell)

    raw_rep = _lr_stats_or_none(payload, "raw_dot_lr_rep_active",
                                "raw_dot_lr_rep_inactive", ell)
    raw_diff = _lr_stats_or_none(payload, "raw_dot_lr_diff_active",
                                 "raw_dot_lr_diff_inactive", ell)
    raw_deriv = _lr_stats_or_none(payload, "raw_dot_lr_deriv_active",
                                  "raw_dot_lr_deriv_inactive", ell)
    raw_deriv10 = _lr_stats_or_none(payload, "raw_dot_lr_deriv10_active",
                                    "raw_dot_lr_deriv10_inactive", ell)
    raw_deriv50 = _lr_stats_or_none(payload, "raw_dot_lr_deriv50_active",
                                    "raw_dot_lr_deriv50_inactive", ell)
    raw_lg = _lr_stats_or_none(payload, "raw_dot_lr_loss_grad_active",
                               "raw_dot_lr_loss_grad_inactive", ell)
    raw_ntk_grad = _lr_ntk_pair_curves(payload, "raw_dot_lr_ntk_grad_pair", ell)

    cos_titles = [
        (rep, steps_np, r"representation $h(t)$"),
        (diff, steps_np,
         r"$\Delta$ checkpoint  $h(t) - h(t_\mathrm{prev\ probe})$"),
        (deriv, deriv_steps, r"$\Delta_t$ update  $h(t) - h(t{-}1)$"),
        (deriv10, deriv10_steps,
         r"$\Delta_{10}$ window  $h(t) - h(t{-}10)$"),
        (deriv50, deriv50_steps,
         r"$\Delta_{50}$ window  $h(t) - h(t{-}50)$"),
        (lg, steps_np, r"activation descent  $-\partial L/\partial h$"),
    ]
    for j, (stats, xs, title) in enumerate(cos_titles):
        if stats is None or xs is None:
            axes[2, j].set_axis_off()
            axes[2, j].set_title(f"{title} unavailable")
            continue
        mA, sA, mI, sI = stats
        _draw_panel(axes[2, j], xs, mA, sA, mI, sI, title)
    if ntk_grad is not None:
        _draw_ntk_pair_panel(axes[2, 6], steps_np, ntk_grad,
                             r"pair NTK loss-gradient (logreg $w_c$)")
    else:
        axes[2, 6].set_axis_off()
        axes[2, 6].set_title("logreg loss-gradient pair NTK unavailable")
    ntk_eig = _lr_ntk_pair_curves(payload, "dot_lr_ntk_pair", ell)
    if ntk_eig is not None:
        _draw_ntk_pair_panel(axes[2, 7], steps_np, ntk_eig,
                             r"pair NTK eigentest (logreg $w_c$)")
    else:
        axes[2, 7].set_axis_off()
        axes[2, 7].set_title("logreg NTK eigentest unavailable")
    axes[2, 8].set_axis_off()
    axes[2, 0].set_ylabel(r"$\cos(w_c^{\mathrm{LR}},\;X(t))$"
                          "  (mean $\\pm$ std over K classes)")

    raw_panels = [
        (raw_rep, steps_np, r"raw dot  $h(t)$"),
        (raw_diff, steps_np,
         r"raw dot  $h(t) - h(t_\mathrm{prev\ probe})$"),
        (raw_deriv, deriv_steps, r"raw dot  $h(t) - h(t{-}1)$"),
        (raw_deriv10, deriv10_steps, r"raw dot  $h(t) - h(t{-}10)$"),
        (raw_deriv50, deriv50_steps, r"raw dot  $h(t) - h(t{-}50)$"),
        (raw_lg, steps_np, r"raw dot  $-\partial L/\partial h$"),
    ]
    for j, (stats, xs, title) in enumerate(raw_panels):
        if stats is None or xs is None:
            axes[3, j].set_axis_off()
            axes[3, j].set_title(f"{title} unavailable")
            continue
        mA, sA, mI, sI = stats
        _draw_panel(axes[3, j], xs, mA, sA, mI, sI, title)
    if raw_ntk_grad is not None:
        _draw_ntk_pair_panel(axes[3, 6], steps_np, raw_ntk_grad,
                             r"raw dot loss-gradient pair NTK (logreg $w_c$)")
    else:
        axes[3, 6].set_axis_off()
        axes[3, 6].set_title("logreg raw loss-gradient pair NTK unavailable")
    raw_ntk_eig = _lr_ntk_pair_curves(payload, "raw_dot_lr_ntk_pair", ell)
    if raw_ntk_eig is not None:
        _draw_ntk_pair_panel(axes[3, 7], steps_np, raw_ntk_eig,
                             r"raw dot pair NTK eigentest (logreg $w_c$)")
    else:
        axes[3, 7].set_axis_off()
        axes[3, 7].set_title("logreg raw NTK eigentest unavailable")
    axes[3, 8].set_axis_off()
    axes[3, 0].set_ylabel(r"$\langle w_c^{\mathrm{LR}},\;X(t)\rangle$"
                          "  (mean $\\pm$ std over K classes)")


def _plot_per_class_loss_grad(payload, ell, d_star, p_star, steps_np, out_dir):
    """K per-class curves for -dL/dh alignment with within-class sample std."""
    needed = ("dot_loss_grad_active", "dot_loss_grad_active_std",
              "dot_loss_grad_inactive", "dot_loss_grad_inactive_std")
    if not all(k in payload and ell in payload[k] for k in needed):
        print("per-class loss-grad std not in payload; skipping per-class plot")
        return
    cos_a = payload["dot_loss_grad_active"][ell][:, d_star, p_star, :]
    cos_a_std = payload["dot_loss_grad_active_std"][ell][:, d_star, p_star, :]
    cos_i = payload["dot_loss_grad_inactive"][ell][:, d_star, p_star, :]
    cos_i_std = payload["dot_loss_grad_inactive_std"][ell][:, d_star, p_star, :]
    raw_a = payload["raw_dot_loss_grad_active"][ell][:, d_star, p_star, :]
    raw_a_std = payload["raw_dot_loss_grad_active_std"][ell][:, d_star, p_star, :]
    raw_i = payload["raw_dot_loss_grad_inactive"][ell][:, d_star, p_star, :]
    raw_i_std = payload["raw_dot_loss_grad_inactive_std"][ell][:, d_star, p_star, :]

    K = cos_a.shape[1]
    T = cos_a.shape[0]
    x = steps_np[:T]
    cmap = plt.get_cmap("tab20" if K <= 20 else "viridis")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = [
        (axes[0, 0], cos_a, cos_a_std,
         r"cosine, active samples  $\cos(w_c^{\mathrm{final}}, -\partial L/\partial h_c(x))$,  $y_\ell(x)=c$"),
        (axes[0, 1], cos_i, cos_i_std,
         r"cosine, inactive samples  ($y_\ell(x)\neq c$)"),
        (axes[1, 0], raw_a, raw_a_std,
         r"raw dot, active samples"),
        (axes[1, 1], raw_i, raw_i_std,
         r"raw dot, inactive samples"),
    ]
    for ax, mean, std, title in panels:
        for c in range(K):
            color = cmap(c % cmap.N)
            m = mean[:, c].numpy()
            s = std[:, c].numpy()
            ax.plot(x, m, color=color, lw=1.2, alpha=0.9)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.12)
        ax.axhline(0.0, color="grey", ls="--", lw=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("training step")
    axes[0, 0].set_ylabel(r"$\cos$  (per-class mean $\pm$ within-class sample std)")
    axes[1, 0].set_ylabel(r"$\langle\cdot\rangle$  (per-class mean $\pm$ within-class sample std)")

    fig.suptitle(
        f"Layer {ell}, cell (d={d_star}, p={p_star})  —  "
        r"per-latent alignment of $-\partial L/\partial h$ with $w_c^{\mathrm{final}}$  "
        f"(K={K} classes; each curve = one class)"
    )
    plot_path = out_dir / f"probe_dot_per_class_{ell}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved per-class plot -> {plot_path}")


if __name__ == "__main__":
    main()
