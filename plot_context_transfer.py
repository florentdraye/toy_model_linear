"""Create raw pilot figures from per-latent context-transfer CSV files."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


INTS = {"seed", "step", "latent", "eval_id", "count", "background", "repeat", "subset", "m", "source_id"}


def load(root, name):
    rows = []
    for path in sorted(root.glob(f"latent_*/{name}.csv")):
        with path.open() as f:
            for row in csv.DictReader(f):
                for key, value in list(row.items()):
                    if key == "subset_slots" or key in ("stage", "source"): continue
                    row[key] = int(value) if key in INTS else float(value)
                rows.append(row)
    if not rows: raise FileNotFoundError(name)
    return rows


def groups(rows, keys):
    out = defaultdict(list)
    for row in rows: out[tuple(row[k] for k in keys)].append(row)
    return out


def export(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)


def colors(root):
    meta = [json.loads(p.read_text()) for p in root.glob("latent_*/metadata.json")]
    if not meta or not all(x["complete"] for x in meta): raise ValueError("incomplete measurements")
    freq = {x["latent"]: x["frequency"] for x in meta}
    return freq, LogNorm(min(freq.values()), max(freq.values())), plt.colormaps["viridis"]


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def count_plot(rows, out, freq, norm, cmap):
    # The smaller saved step is the declared linear-regime actual-update curve.
    eta = min(r["eta"] for r in rows); data = [r for r in rows if r["eta"] == eta]
    stages = [x for x in ("before", "during", "after") if any(r["stage"] == x for r in data)]
    fig, axes = plt.subplots(2, len(stages), figsize=(4.2*len(stages), 6.6), sharex=True)
    for col, stage in enumerate(stages):
        d = [r for r in data if r["stage"] == stage]
        for row_ix, field in enumerate(("actual_loss_decrease", "predicted_loss_decrease")):
            ax = axes[row_ix, col]
            for latent in sorted(freq):
                z = groups([r for r in d if r["latent"] == latent], ("count",))
                x = np.array(sorted(k[0] for k in z)); vals = [np.array([r[field] for r in z[(n,)]]) for n in x]
                mean, sd = np.array([v.mean() for v in vals]), np.array([v.std() for v in vals])
                color = cmap(norm(freq[latent])); ax.plot(x, mean, color=color, lw=1.25)
                ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.055, linewidth=0)
            ax.axhline(0, color=".5", lw=.7); ax.grid(alpha=.18); ax.set_title(f"{stage} acquisition")
            if col == 0: ax.set_ylabel("Actual loss decrease" if row_ix == 0 else "First-order prediction")
            if row_ix == 1: ax.set_xlabel("Examples containing latent in batch (n=128)")
    fig.suptitle(f"Transfer versus exact latent count — seed 46, η={eta:g}\n"
                 "Lines: context means; translucent bands: SD over source contexts")
    fig.tight_layout(); save(fig, out/"count_transfer")


def variation_plot(rows, out, freq, norm, cmap):
    summary = []
    for key, z in groups(rows, ("seed", "step", "latent", "eval_id", "background")).items():
        q = np.array([r["q"] for r in z]); second = np.mean(q*q)
        summary.append(dict(zip(("seed", "step", "latent", "eval_id", "background"), key),
                            mean_q=q.mean(), second_moment=second, variance=q.var(),
                            V=q.var()/second if second > 1e-24 else np.nan))
    export(out/"context_variation_summary.csv", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for latent in sorted(freq):
        by = groups([r for r in summary if r["latent"] == latent], ("step",)); steps = sorted(k[0] for k in by)
        v = [np.nanmean([r["V"] for r in by[(s,)]]) for s in steps]
        q = [np.mean([r["mean_q"] for r in by[(s,)]]) for s in steps]
        color = cmap(norm(freq[latent])); axes[0].plot(steps, v, color=color, lw=1.25)
        axes[1].plot(steps, q, color=color, lw=1.25)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Relative context variation V")
    axes[1].axhline(0, color=".5", lw=.7); axes[1].set_ylabel("Mean extra benefit  gᵢᵀ[T(Bₛ)−T(B₀)]")
    for ax in axes: ax.set_xlabel("Training step"); ax.grid(alpha=.18)
    fig.suptitle("Context identity at fixed latent count — seed 46, m=64 of n=128")
    fig.tight_layout(); save(fig, out/"context_variation")


def geometry_plot(agreement, alignment, out, freq, norm, cmap):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for data, ax, shared_name, title, ylabel in (
        (agreement, axes[0], "shared_latent", "Contribution agreement", "cos(Tᵢⱼ, mean reference contribution)"),
        (alignment, axes[1], "shared_latent_batch", "Alignment with useful learning", "cos(gᵢ, Tᵢ(B))")):
        for latent in sorted(freq):
            z = [r for r in data if r["latent"] == latent and r["source"] == shared_name]
            by = groups(z, ("step",)); steps = sorted(k[0] for k in by)
            mean = [np.nanmean([r["cosine"] for r in by[(s,)]]) for s in steps]
            ax.plot(steps, mean, color=cmap(norm(freq[latent])), lw=1.2)
        control = groups([r for r in data if r["source"] == "no_shared_latent"], ("step",)); steps = sorted(k[0] for k in control)
        ax.plot(steps, [np.nanmean([r["cosine"] for r in control[(s,)]]) for s in steps],
                color=".35", ls="--", lw=1.6, label="no shared latent")
        ax.axhline(0, color=".5", lw=.7); ax.set_ylim(-1, 1); ax.grid(alpha=.18)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("Training step"); ax.legend(frameon=False)
    fig.suptitle("Geometry of cross-context learning — raw checkpoints, seed 46")
    fig.tight_layout(); save(fig, out/"transfer_geometry")


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("root", type=Path); p.add_argument("--out-dir", type=Path)
    a = p.parse_args(); out = a.out_dir or a.root/"figures"; out.mkdir(parents=True, exist_ok=True)
    freq, norm, cmap = colors(a.root)
    tables = {name: load(a.root, name) for name in
              ("count_transfer", "context_variation", "contribution_agreement", "useful_alignment")}
    for name, rows in tables.items(): export(out/f"{name}.csv", rows)
    count_plot(tables["count_transfer"], out, freq, norm, cmap)
    variation_plot(tables["context_variation"], out, freq, norm, cmap)
    geometry_plot(tables["contribution_agreement"], tables["useful_alignment"], out, freq, norm, cmap)
    summary = {"latents": len(freq), "steps": sorted({r["step"] for r in tables["context_variation"]}),
               "rows": {k: len(v) for k, v in tables.items()}}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
