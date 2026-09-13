"""Display calibration-selected reference-mean locations and held-out curves."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, BoundaryNorm

from plot_emergence import smooth


def plot(paths, out, smooth_window=3):
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError('smooth-window must be a positive odd number')
    data = [json.loads(Path(p).read_text()) for p in paths]
    ref = data[0]
    for d in data[1:]:
        if d.get('model_transform') != ref.get('model_transform'):
            raise ValueError('cannot pool different model transformations')
        for key in ('latents', 'frequency', 'cells', 'reference', 'model_config',
                    'evaluation_bank_sha256', 'direction_estimator', 'selection'):
            if d[key] != ref[key]:
                raise ValueError(f'cannot pool different {key}')
    if len({d['config']['seed'] for d in data}) != len(data):
        raise ValueError('duplicate seeds')
    steps = sorted(set.intersection(*[{r['step'] for r in d['history']} for d in data]))
    rows = [[{r['step']: r for r in d['history']}[t] for t in steps] for d in data]
    arrays = {key: np.array([[r[key] for r in group] for group in rows])
              for key in ('steer_raw', 'gain_raw', 'patch_raw', 'random_raw', 'fixed_block1_steer')}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    freq = np.array(ref['frequency'])
    norm = LogNorm(freq.min()*.999, freq.max()*1.001)
    cmap = LinearSegmentedColormap.from_list('frequency', plt.get_cmap('viridis_r')(np.linspace(.18, .98, 256)))
    colors = [cmap(norm(p)) for p in freq]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#293344', 'savefig.dpi': 220})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True, layout='constrained')
    for ax, key, title in zip(axes, ('fixed_block1_steer', 'steer_raw', 'gain_raw'),
                              ('Fixed block 1 · tokens 2–5', 'Best training-selected location', 'Generalization')):
        mean = np.clip(arrays[key], 0, 1).mean(0)
        for j, color in enumerate(colors):
            ax.plot(steps, mean[:, j], '.', ms=2, color=color, alpha=.25)
            ax.plot(steps, smooth(mean[:, j], smooth_window), color=color, lw=1.5)
        ax.set(title=title, xlabel='Training steps', ylim=(-.025, 1.025))
        ax.grid(axis='y', color='#e8ebef', lw=.6)
    axes[0].set_ylabel('Target fidelity (Brier skill, floored at 0)')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.018, pad=.02)
    cb.set_label('Latent training frequency')
    status = 'complete' if all(d.get('complete') for d in data) else 'partial scan'
    seed_label = f"seed {ref['config']['seed']}" if len(data) == 1 else f'{len(data)} seeds'
    strengths = ref.get('alphas', [1.])
    strength_label = 'strength 1' if strengths == [1.] else f'training-selected strength from {strengths}'
    transform = ref.get('model_transform', {})
    weight_label = (f" · model-weight EMA {transform['decay']}"
                    if transform.get('decay', 0) else '')
    fig.suptitle(f"Width {ref['model_config']['d_model']} · {len(freq)} latents · {seed_label} · {status}\n"
                 f'Full-support target − reference means · {strength_label} · fixed held-out examples{weight_label}', fontsize=12)
    fig.savefig(out / 'best_location_steering.png')
    fig.savefig(out / 'best_location_steering.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for key, label, color in [('fixed_block1_steer', 'Fixed block 1', '#a8aeb8'),
                              ('steer_raw', 'Selected mean edit', '#a9682f'),
                              ('patch_raw', 'Exact patch at selected site', '#4e8e82'),
                              ('gain_raw', 'Generalization', '#273952')]:
        seed_means = arrays[key].mean(2)
        m = seed_means.mean(0)
        ax.plot(steps, m, color=color, lw=2, label=label)
        if len(data) > 1:
            se = seed_means.std(0, ddof=1)/len(data)**.5
            ax.fill_between(steps, m-1.96*se, m+1.96*se, color=color, alpha=.13)
    ax.axhline(0, color='#999', lw=.7)
    ax.set(xlabel='Training steps', ylabel='Raw Brier skill · mean across latents',
           title='Unclipped, unsmoothed held-out scores')
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / 'raw_selected_steering.png')
    plt.close(fig)
    if len(steps) > 1:
        depth = np.array([[[c['depth'] for c in r['choices']] for r in g] for g in rows])
        names = list(dict.fromkeys(c['site'] for c in ref['cells']))
        site = np.array([[[names.index(c['site']) for c in r['choices']] for r in g] for g in rows])
        fig, axes = plt.subplots(len(data), 2, figsize=(12, 3.5*len(data)), squeeze=False,
                                 sharex=True, sharey=True, layout='constrained')
        for j, (arr, labels) in enumerate(((depth, ['emb', '1', '2', '3', '4', '5', '6']), (site, names))):
            cm = plt.get_cmap('tab10', len(labels))
            nm = BoundaryNorm(np.arange(len(labels)+1)-.5, cm.N)
            for s, d in enumerate(data):
                im = axes[s, j].pcolormesh(steps, np.arange(len(freq)), arr[s].T,
                                            shading='nearest', cmap=cm, norm=nm)
                axes[s, j].set_title(f"Seed {d['config']['seed']} · {'block' if j == 0 else 'token site'}")
                axes[s, j].set_yticks(range(len(freq)), ref['latents'], fontsize=7)
                if j == 0:
                    axes[s, j].set_ylabel('Latent (frequency order)')
                if s == len(data)-1:
                    axes[s, j].set_xlabel('Training steps')
            cb = fig.colorbar(im, ax=axes[:, j], ticks=range(len(labels)), fraction=.025)
            cb.ax.set_yticklabels(labels)
        fig.suptitle('Locations selected on training pairs at each checkpoint')
        fig.savefig(out / 'selected_locations.png')
        plt.close(fig)
    (out / 'figure_notes.txt').write_text(
        f"Sources: {paths}\nEvaluation-bank SHA256: {ref['evaluation_bank_sha256']}\n"
        "Location argmax uses only training calibration pairs; held-out examples never select locations.\n"
        "All cells: 7 depths (embedding plus 6 blocks), 6 individual tokens, suffix, all tokens.\n"
        f"Strength grid: {strengths}; any strength selection uses training calibration only.\n"
        f"Bounded panel floors each seed before averaging; {smooth_window}-checkpoint display window with raw dots "
        "(window 1 means no curve smoothing).\n"
        "Raw scores are unsmoothed and retain negatives; bands +/-1.96 SE across model seeds.\n")
    print(f'Location figures: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('histories', nargs='+', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--smooth-window', type=int, default=3,
                   help='positive odd display window; 1 retains raw checkpoint values')
    a = p.parse_args()
    plot(a.histories, a.out_dir, a.smooth_window)
