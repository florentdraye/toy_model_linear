"""Stream per-latent transfer results into ten-seed summaries and figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def read(path):
    with path.open() as f: yield from csv.DictReader(f)


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)


def aggregate(roots, out):
    count, variation, agreement, alignment = [], [], [], []
    frequencies = {}
    for root in roots:
        for directory in sorted(root.glob("latent_*")):
            meta = json.loads((directory/"metadata.json").read_text())
            if not meta["complete"]: raise ValueError(f"incomplete {directory}")
            seed, latent = meta["seed"], meta["latent"]; frequencies[latent] = meta["frequency"]
            g = defaultdict(lambda: [[], []])
            for r in read(directory/"count_transfer.csv"):
                key = (r["stage"], int(r["count"]), float(r["eta"]))
                g[key][0].append(float(r["actual_loss_decrease"])); g[key][1].append(float(r["predicted_loss_decrease"]))
            for (stage, n, eta), (actual, predicted) in g.items():
                count.append(dict(seed=seed, latent=latent, stage=stage, count=n, eta=eta,
                                  actual_mean=np.mean(actual), actual_context_sd=np.std(actual),
                                  predicted_mean=np.mean(predicted), predicted_context_sd=np.std(predicted)))
            raw = defaultdict(list)
            for r in read(directory/"context_variation.csv"):
                raw[(int(r["step"]), int(r["eval_id"]), int(r["background"]))].append(float(r["q"]))
            by_step = defaultdict(lambda: [[], []])
            for (step, _, _), q0 in raw.items():
                q = np.asarray(q0); second = np.mean(q*q)
                if second > 1e-24: by_step[step][0].append(q.var()/second)
                by_step[step][1].append(q.mean())
            for step, (v, q) in by_step.items():
                variation.append(dict(seed=seed, latent=latent, step=step,
                                      V_mean=np.mean(v) if v else np.nan, mean_extra_benefit=np.mean(q)))
            for filename, destination in (("contribution_agreement.csv", agreement), ("useful_alignment.csv", alignment)):
                values = defaultdict(list)
                for r in read(directory/filename): values[(int(r["step"]), r["source"])].append(float(r["cosine"]))
                for (step, source), x in values.items():
                    destination.append(dict(seed=seed, latent=latent, step=step, source=source,
                                            cosine_mean=np.nanmean(x), cosine_context_sd=np.nanstd(x)))
    write(out/"count_transfer_seed_summary.csv", count); write(out/"context_variation_seed_summary.csv", variation)
    write(out/"contribution_agreement_seed_summary.csv", agreement); write(out/"useful_alignment_seed_summary.csv", alignment)
    return count, variation, agreement, alignment, frequencies


def group(rows, keys):
    result = defaultdict(list)
    for r in rows: result[tuple(r[k] for k in keys)].append(r)
    return result


def save(fig, out, name):
    fig.savefig(out/f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out/f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def plots(count, variation, agreement, alignment, freq, out, seeds):
    cmap, norm = plt.colormaps["viridis"], LogNorm(min(freq.values()), max(freq.values()))
    eta = min(r["eta"] for r in count); data = [r for r in count if r["eta"] == eta]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), sharex=True)
    for col, stage in enumerate(("before", "during", "after")):
        for row_ix, field in enumerate(("actual_mean", "predicted_mean")):
            ax = axes[row_ix, col]
            for latent in sorted(freq):
                z = [r for r in data if r["stage"] == stage and r["latent"] == latent]
                by = group(z, ("count",)); x = np.array(sorted(k[0] for k in by))
                mean = np.array([np.mean([r[field] for r in by[(n,)]]) for n in x])
                sd = np.array([np.std([r[field] for r in by[(n,)]]) for n in x])
                color = cmap(norm(freq[latent])); ax.plot(x, mean, color=color, lw=1.25)
                ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.06, linewidth=0)
            ax.axhline(0, color=".5", lw=.7); ax.grid(alpha=.18); ax.set_title(f"{stage} acquisition")
            if col == 0: ax.set_ylabel("Actual loss decrease" if row_ix == 0 else "First-order prediction")
            if row_ix == 1: ax.set_xlabel("Latent-present examples in batch (n=128)")
    fig.suptitle(f"Transfer versus exact latent count — {len(seeds)}-seed mean, η={eta:g}\nBands: SD across model seeds")
    fig.tight_layout(); save(fig, out, "count_transfer_multiseed")

    # Only the regular 500-step grid has all seeds; acquisition-specific extra steps are excluded.
    regular = [r for r in variation if r["step"] % 500 == 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for latent in sorted(freq):
        z = [r for r in regular if r["latent"] == latent]; by = group(z, ("step",)); steps = sorted(k[0] for k in by)
        for ax, field in zip(axes, ("V_mean", "mean_extra_benefit")):
            mean = np.array([np.nanmean([r[field] for r in by[(s,)]]) for s in steps])
            sd = np.array([np.nanstd([r[field] for r in by[(s,)]]) for s in steps])
            color = cmap(norm(freq[latent])); ax.plot(steps, mean, color=color, lw=1.2)
            ax.fill_between(steps, mean-sd, mean+sd, color=color, alpha=.04, linewidth=0)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Relative context variation V")
    axes[1].axhline(0, color=".5", lw=.7); axes[1].set_yscale("symlog", linthresh=1.)
    axes[1].set_ylabel("Mean extra benefit (symmetric log scale)")
    for ax in axes: ax.set_xlabel("Training step"); ax.grid(alpha=.18)
    fig.suptitle(f"Context identity at fixed count — {len(seeds)}-seed mean; m=64 of n=128")
    fig.tight_layout(); save(fig, out, "context_variation_multiseed")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for data0, ax, shared, title, ylabel in (
        (agreement, axes[0], "shared_latent", "Contribution agreement", "cos(Tᵢⱼ, mean Tᵢℓ)"),
        (alignment, axes[1], "shared_latent_batch", "Alignment with useful learning", "cos(gᵢ, Tᵢ(B))")):
        regular0 = [r for r in data0 if r["step"] % 500 == 0]
        for latent in sorted(freq):
            z = [r for r in regular0 if r["latent"] == latent and r["source"] == shared]; by = group(z, ("step",)); steps = sorted(k[0] for k in by)
            mean = [np.nanmean([r["cosine_mean"] for r in by[(s,)]]) for s in steps]
            ax.plot(steps, mean, color=cmap(norm(freq[latent])), lw=1.2)
        control = group([r for r in regular0 if r["source"] == "no_shared_latent"], ("step",)); steps = sorted(k[0] for k in control)
        ax.plot(steps, [np.nanmean([r["cosine_mean"] for r in control[(s,)]]) for s in steps],
                color=".35", ls="--", lw=1.6, label="no shared latent")
        ax.axhline(0, color=".5", lw=.7); ax.set_ylim(-1, 1); ax.grid(alpha=.18)
        ax.set_title(title); ax.set_xlabel("Training step"); ax.set_ylabel(ylabel); ax.legend(frameon=False)
    fig.suptitle(f"Geometry of cross-context learning — {len(seeds)}-seed mean")
    fig.tight_layout(); save(fig, out, "transfer_geometry_multiseed")


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("roots", type=Path, nargs="+"); p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(str(root).rsplit("s", 1)[-1]) for root in a.roots})
    if len(seeds) != len(a.roots): raise ValueError("duplicate seed roots")
    count, variation, agreement, alignment, freq = aggregate(a.roots, a.out_dir)
    plots(count, variation, agreement, alignment, freq, a.out_dir, seeds)
    summary = {"seeds": seeds, "latents": sorted(freq), "eta_linear": min(r["eta"] for r in count),
               "rows": {"count": len(count), "variation": len(variation), "agreement": len(agreement), "alignment": len(alignment)}}
    (a.out_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
