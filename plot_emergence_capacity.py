"""Compare model capacities on exactly the same fixed latent/pair bank."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from plot_emergence import smooth


def compare(small_paths, large_paths, out, window=3):
    groups = [[json.loads(Path(p).read_text()) for p in paths]
              for paths in (small_paths, large_paths)]
    ref = groups[0][0]
    for group in groups:
        if len({d['config']['seed'] for d in group}) != len(group):
            raise ValueError('duplicate seeds')
        for d in group:
            for key in ('latents', 'frequency', 'reference', 'positions', 'graph_config',
                        'metric', 'direction_estimator', 'evaluation_bank_sha256',
                        'training_support_sha256'):
                if d[key] != ref[key]:
                    raise ValueError(f'capacity comparison differs in {key}')
            for key in ('data_seed', 'graph_layer', 'steer_depth', 'site', 'lr',
                        'weight_decay', 'batch_size', 'train_frac', 'frequency_ratio',
                        'direction_method'):
                if d['config'][key] != ref['config'][key]:
                    raise ValueError(f'capacity comparison differs in {key}')
            for key, value in ref['model_config'].items():
                if key not in ('d_model', 'd_ff') and d['model_config'][key] != value:
                    raise ValueError(f'architecture differs beyond width: {key}')
        if any(d['model_config'] != group[0]['model_config'] for d in group):
            raise ValueError('cannot pool different model sizes')
    if sorted(d['config']['seed'] for d in groups[0]) != sorted(d['config']['seed'] for d in groups[1]):
        raise ValueError('capacity comparison needs the same seed set')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    freq = np.array(ref['frequency'])
    norm = LogNorm(freq.min() * .999, freq.max() * 1.001)
    cmap = LinearSegmentedColormap.from_list('frequency', plt.get_cmap('viridis_r')(np.linspace(.18, .98, 256)))
    colors = [cmap(norm(p)) for p in freq]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#293344', 'savefig.dpi': 220})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True,
                             layout='constrained')
    curves, rows = [], []
    for i, group in enumerate(groups):
        shared = sorted(set.intersection(*[{h['step'] for h in d['history']} for d in group]))
        if not shared:
            raise ValueError('no shared checkpoints')
        arr = {key: np.array([[{h['step']: h for h in d['history']}[t][key]
                               for t in shared] for d in group])
               for key in ('steer_raw', 'gain_raw', 'patch_raw', 'random_raw')}
        curves.append((shared, arr))
        width = group[0]['model_config']['d_model']
        for j, key in enumerate(('steer_raw', 'gain_raw')):
            ax = axes[i, j]
            y = np.clip(arr[key], 0, 1).mean(0)
            for k, color in enumerate(colors):
                ax.plot(shared, y[:, k], '.', color=color, ms=2, alpha=.25)
                ax.plot(shared, smooth(y[:, k], window), color=color, lw=1.4, alpha=.85)
            ax.set_title(f"Width {width} · {'Steering' if j == 0 else 'Generalization'}")
            ax.axhline(.5, color='#a8afba', ls=':', lw=.8)
            ax.grid(axis='y', color='#e8ebef', lw=.6)
            ax.set_ylim(-.025, 1.025)
            if i == 1:
                ax.set_xlabel('Training steps')
            if j == 0:
                ax.set_ylabel('Target fidelity (Brier skill, floored at 0)')
        # Raw, equal-weight latent averages; no flooring or smoothing in table.
        for d in group:
            for h in d['history']:
                rows.append({'width': width, 'seed': d['config']['seed'], 'step': h['step'],
                             **{k: float(np.mean(h[k])) for k in arr},
                             'worst_latent_steer': min(h['steer_raw'])})
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        fraction=.025, pad=.025)
    cbar.set_label('Latent training frequency')
    status = 'complete' if all(d.get('complete') for g in groups for d in g) else 'partial larger run'
    fig.suptitle(f"Same {len(freq)} latents and held-out pairs · {len(groups[0])} seeds · {status}\n"
                 f"Target − reference means · block {ref['config']['steer_depth']} · tokens {ref['positions']}", fontsize=12)
    fig.savefig(out / 'capacity_comparison.png')
    fig.savefig(out / 'capacity_comparison.pdf')
    plt.close(fig)
    # Preserve raw averages and seed variation alongside the familiar bounded panels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True, layout='constrained')
    for group, (steps, arr), color in zip(groups, curves, ('#648ca6', '#bf6635')):
        for ax, key in zip(axes, ('steer_raw', 'gain_raw')):
            seed_means = arr[key].mean(2)
            mean = seed_means.mean(0)
            ax.plot(steps, mean, color=color, lw=2, label=f"Width {group[0]['model_config']['d_model']}")
            if len(group) > 1:
                se = seed_means.std(0, ddof=1) / len(group) ** .5
                ax.fill_between(steps, mean - 1.96*se, mean + 1.96*se, color=color, alpha=.15)
    for ax, title in zip(axes, ('Steering', 'Generalization')):
        ax.set(title=title, xlabel='Training steps', ylabel='Raw Brier skill · mean across 32 latents')
        ax.axhline(0, color='#aaa', lw=.7)
        ax.legend(frameon=False)
    fig.suptitle('Equal-weight latent averages · bands show variation across model seeds')
    fig.savefig(out / 'capacity_raw_average.png')
    plt.close(fig)
    with (out / 'capacity_scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / 'figure_notes.txt').write_text(
        f"Sources: {small_paths}; {large_paths}\n"
        f"Fixed evaluation-bank SHA256: {ref['evaluation_bank_sha256']}\n"
        f"32 curves selected by frequency rank before observing larger-model outcomes.\n"
        f"Bounded panels: per-seed flooring then averaging; {window}-checkpoint smoothing; raw dots.\n"
        "Raw average: no flooring or smoothing; equal weight per latent; bands +/-1.96 SE across seeds.\n"
        "Width-128 snapshots end at 10,000 steps; width-512 was scheduled for 20,000.\n"
        "Use the shared training-step range for comparisons of capacity at equal training exposure.\n")
    print(f'Capacity comparison: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small', nargs='+', type=Path, required=True)
    p.add_argument('--large', nargs='+', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    compare(a.small, a.large, a.out_dir)
