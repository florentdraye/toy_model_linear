"""Compare completed capacity runs using the same unit-strength location search."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap


def plot(paths, out):
    data = [json.loads(Path(p).read_text()) for p in paths]
    ref = data[0]
    groups = {}
    for d in data:
        if not d.get('complete'):
            raise ValueError('this summary requires completed location scans')
        if d.get('alphas', [1.]) != [1.]:
            raise ValueError('unit-strength comparison only')
        for key in ('latents', 'frequency', 'cells', 'reference', 'evaluation_bank_sha256',
                    'direction_estimator', 'selection'):
            if d[key] != ref[key]:
                raise ValueError(f'comparison differs in {key}')
        for key, value in ref['model_config'].items():
            if key not in ('d_model', 'd_ff') and d['model_config'][key] != value:
                raise ValueError(f'architecture differs beyond width: {key}')
        groups.setdefault(d['model_config']['d_model'], []).append(d)
    seeds = sorted(d['config']['seed'] for d in next(iter(groups.values())))
    if len(seeds) != len(set(seeds)):
        raise ValueError('duplicate seeds')
    if any(sorted(d['config']['seed'] for d in g) != seeds for g in groups.values()):
        raise ValueError('all widths need the same seed set')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    freq = np.array(ref['frequency'])
    norm = LogNorm(freq.min()*.999, freq.max()*1.001)
    cmap = LinearSegmentedColormap.from_list('frequency', plt.get_cmap('viridis_r')(np.linspace(.18, .98, 256)))
    colors = [cmap(norm(p)) for p in freq]
    lower = {key: min(-.025, min(x for d in data for r in d['history'] for x in r[key])-.025)
             for key in ('steer_raw', 'gain_raw')}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#293344', 'savefig.dpi': 220})
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 3.35*len(groups)),
                             sharex=True, squeeze=False, layout='constrained')
    records = []
    for i, (width, group) in enumerate(sorted(groups.items())):
        steps = sorted(set.intersection(*[{r['step'] for r in d['history']} for d in group]))
        rows = [[{r['step']: r for r in d['history']}[t] for t in steps] for d in group]
        for j, key in enumerate(('steer_raw', 'gain_raw')):
            arr = np.array([[r[key] for r in g] for g in rows])
            mean = arr.mean(0)
            ax = axes[i, j]
            for k, color in enumerate(colors):
                ax.plot(steps, mean[:, k], color=color, lw=1.25, alpha=.8)
            ax.plot(steps, mean.mean(1), color='#202b3c', lw=2.2, label='Mean across latents')
            ax.axhline(0, color='#888', lw=.8, ls=':')
            ax.set_title(f"Width {width} · {'Selected mean steering' if j == 0 else 'Generalization'}")
            ax.set_ylim(lower[key], 1.025)
            ax.set_ylabel('Raw Brier skill')
            ax.grid(axis='y', color='#e8ebef', lw=.6)
            if i == len(groups)-1:
                ax.set_xlabel('Training steps')
            ax.text(.98, .08 if j == 1 else .05, f'Final mean: {mean[-1].mean():.3f}',
                    transform=ax.transAxes, ha='right', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        for d in group:
            for r in d['history']:
                records.append(dict(width=width, seed=d['config']['seed'], step=r['step'],
                    generalization=float(np.mean(r['gain_raw'])),
                    fixed_block1=float(np.mean(r['fixed_block1_steer'])),
                    selected_steering=float(np.mean(r['steer_raw'])),
                    selected_patch=float(np.mean(r['patch_raw'])),
                    worst_latent_steering=min(r['steer_raw'])))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.022, pad=.02)
    cb.set_label('Latent training frequency')
    fig.suptitle(f"Same {len(freq)} latents and fixed held-out pairs · {len(seeds)} seeds per width\n"
                 'Difference of means · strength 1 · locations selected on training pairs', fontsize=13)
    fig.savefig(out / 'capacity_best_location.png')
    fig.savefig(out / 'capacity_best_location.pdf')
    plt.close(fig)
    with (out / 'capacity_best_scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (out / 'capacity_best_notes.txt').write_text(
        f"Sources: {paths}\nEvaluation-bank SHA256: {ref['evaluation_bank_sha256']}\n"
        "Raw, unsmoothed scores; no clipping or per-curve normalization.\n"
        "Colored lines average each latent over model seeds; dark line averages all latents equally.\n"
        "The location search is identical for every capacity and never selects on test examples.\n"
        "Compare at 10,000 steps for equal training exposure; width 256/512 additionally ran to 20,000.\n")
    print(f'Completed capacity summary: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('histories', nargs='+', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    plot(a.histories, a.out_dir)
