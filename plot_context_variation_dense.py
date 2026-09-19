"""Plot unsmoothed dense fixed-count context variation for one latent."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows, raw_out = [], []
    for root in a.roots:
        meta = json.loads((root/"latent_92"/"metadata.json").read_text())
        if not meta["complete"] or meta["latent"] != 92: raise ValueError(root)
        raw = defaultdict(list)
        with (root/"latent_92"/"context_variation.csv").open() as f:
            for r in csv.DictReader(f):
                key = (int(r["step"]), int(r["eval_id"]), int(r["background"]))
                raw[key].append(float(r["q"]))
        by_step = defaultdict(lambda: [[], []])
        for (step, _, _), q0 in raw.items():
            q = np.asarray(q0); second = np.mean(q*q)
            if second > 1e-24: by_step[step][0].append(q.var()/second)
            by_step[step][1].append(q.mean())
        for step, (v, q) in sorted(by_step.items()):
            seed_rows.append(dict(seed=meta["seed"], step=step, latent=92,
                                  V=np.mean(v), mean_q=np.mean(q)))
    fields = list(seed_rows[0])
    with (a.out_dir/"dense_variation_seed_summary.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fields);w.writeheader();w.writerows(seed_rows)
    grouped=defaultdict(list)
    for r in seed_rows: grouped[r["step"]].append(r)
    summary=[]
    for step, z in sorted(grouped.items()):
        summary.append(dict(step=step, V_mean=np.mean([r["V"] for r in z]),
                            V_seed_sd=np.std([r["V"] for r in z]),
                            q_mean=np.mean([r["mean_q"] for r in z]),
                            q_seed_sd=np.std([r["mean_q"] for r in z])))
    with (a.out_dir/"dense_variation_mean.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, summary[0]);w.writeheader();w.writerows(summary)
    x=np.array([r["step"] for r in summary]); vm=np.array([r["V_mean"] for r in summary]);vs=np.array([r["V_seed_sd"] for r in summary])
    qm=np.array([r["q_mean"] for r in summary]);qs=np.array([r["q_seed_sd"] for r in summary])
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharex=True)
    axes[0].plot(x,vm,color="#31688e",lw=1.7);axes[0].fill_between(x,vm-vs,vm+vs,color="#31688e",alpha=.18,linewidth=0)
    axes[0].set_ylim(0,1);axes[0].set_ylabel("Relative context variation V")
    axes[1].plot(x,qm,color="#31688e",lw=1.7);axes[1].fill_between(x,qm-qs,qm+qs,color="#31688e",alpha=.18,linewidth=0)
    axes[1].set_yscale("symlog",linthresh=1.);axes[1].axhline(0,color=".5",lw=.7);axes[1].set_ylabel("Mean extra benefit (symmetric log)")
    for ax in axes:ax.set_xlabel("Training step");ax.grid(alpha=.18)
    fig.suptitle("Highest-frequency target latent 92 — every 100 steps, 10-seed mean\nRaw checkpoint estimates; bands are SD across model seeds")
    fig.tight_layout()
    for suffix in ("png","pdf"):fig.savefig(a.out_dir/f"dense_context_variation.{suffix}",dpi=180 if suffix=="png" else None,bbox_inches="tight")
    print(json.dumps(dict(seeds=len(a.roots),checkpoints=len(summary),latent=92,
                          first=summary[0],last=summary[-1]),indent=2))


if __name__=="__main__":main()
