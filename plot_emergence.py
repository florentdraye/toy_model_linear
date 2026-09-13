"""Publication-style paired panels, raw checkpoints, and uncensored timing table."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap


def crossing(steps, curve, threshold=.5):
    """First persistent raw crossing; no smoothing or per-curve normalization."""
    if curve[0] >= threshold:
        return None, 'left-censored'
    for i in range(1, len(curve) - 2):
        if curve[i - 1] < threshold and np.all(curve[i:i + 3] >= threshold):
            t = steps[i - 1] + ((threshold - curve[i - 1]) /
                (curve[i] - curve[i - 1])) * (steps[i] - steps[i - 1])
            return float(t), 'observed'
    return None, 'not sustained within run'


def smooth(values, window):
    if window <= 1:
        return values
    pad = window // 2
    return np.convolve(np.pad(values, (pad, pad), mode='edge'),
                       np.ones(window) / window, mode='valid')


def plot(paths, out, window=1):
    if window < 1 or window % 2 != 1:
        raise ValueError('smoothing window must be a positive odd number')
    data = [json.loads(Path(p).read_text()) for p in paths]
    ref = data[0]
    if ref.get('metric') != 'target_brier_skill_v1':
        raise ValueError('expected target Brier skill history; older effect-only pilots cannot be pooled')
    for d in data[1:]:
        for key in ('metric', 'latents', 'frequency', 'positions', 'reference', 'graph_config', 'model_config'):
            if d[key] != ref[key]:
                raise ValueError(f'cannot pool differing {key}')
        for key in ('data_seed', 'graph_layer', 'steer_depth', 'site', 'lr', 'weight_decay',
                    'batch_size', 'train_frac', 'fit_pairs', 'eval_pairs', 'frequency_ratio',
                    'direction_method', 'direction_steps', 'direction_lr'):
            if d['config'][key] != ref['config'][key]:
                raise ValueError(f'cannot pool differing {key}')
    seeds = [d['config']['seed'] for d in data]
    if len(set(seeds)) != len(seeds):
        raise ValueError('duplicate seeds would create misleading uncertainty')
    shared = sorted(set.intersection(*[{h['step'] for h in d['history']} for d in data]))
    steps = np.array(shared)
    arrays = {}
    for key in ('gain_raw', 'steer_raw', 'patch_raw', 'random_raw', 'mean_raw', 'accuracy'):
        arrays[key] = np.array([[{h['step']: h for h in d['history']}[t][key]
                                 for t in shared] for d in data])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#b9bdc6', 'axes.labelcolor': '#293344',
                         'text.color': '#293344', 'xtick.color': '#657083',
                         'ytick.color': '#657083', 'savefig.dpi': 220})
    freq = np.array(ref['frequency'])
    norm = LogNorm(min(freq) * .999, max(freq) * 1.001)
    cmap = LinearSegmentedColormap.from_list('frequency', plt.get_cmap('viridis_r')(np.linspace(.18, .98, 256)))
    colors = [cmap(norm(p)) for p in freq]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharex=True, sharey=True,
                             layout='constrained')
    for ax, key, title in zip(axes, ('steer_raw', 'gain_raw'),
                              ('Steering the hidden state', 'Generalizing to unseen trajectories')):
        bounded = np.clip(arrays[key], 0, 1)
        mean = bounded.mean(0)
        for j, color in enumerate(colors):
            ax.plot(steps, mean[:, j], '.', color=color, ms=2, alpha=.28)
            ax.plot(steps, smooth(mean[:, j], window), color=color, lw=1.9)
            if len(data) > 1:
                se = bounded[:, :, j].std(0, ddof=1) / len(data) ** .5
                ax.fill_between(steps, np.clip(mean[:, j] - 1.96 * se, 0, 1),
                                np.clip(mean[:, j] + 1.96 * se, 0, 1), color=color, alpha=.09)
        ax.set(title=title, xlabel='Training steps', ylim=(-.025, 1.025))
        ax.grid(axis='y', color='#e8ebef', linewidth=.6)
        ax.axhline(.5, color='#b0b6c1', ls=':', lw=.8)
    axes[0].set_ylabel('Target fidelity (Brier skill, floored at 0)')
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        fraction=.025, pad=.025)
    cbar.set_label('Latent training frequency')
    c = ref['config']
    status = 'complete' if all(d.get('complete') for d in data) else 'partial run'
    fig.suptitle(f"Graph layer {c['graph_layer']} · transformer block {c['steer_depth']} · "
                 f"{len(data)} seed{'s' if len(data)>1 else ''} · {status}", fontsize=12)
    fig.savefig(out / 'steering_gain.png')
    fig.savefig(out / 'steering_gain.pdf')
    plt.close(fig)
    # Small multiples pair the SAME latent directly and expose controls.
    n = len(freq)
    fig, axs = plt.subplots((n + 3)//4, min(n, 4), figsize=(12, 2.5*((n+3)//4)),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    for j, ax in enumerate(axs.flat):
        if j >= n:
            ax.set_visible(False)
            continue
        for key, label, style, color in [('patch_raw', 'Exact patch', '--', '#959aa4'),
                                         ('mean_raw', 'Mean vector', '-.', '#8c76b7'),
                                         ('random_raw', 'Random direction', ':', '#cb826d'),
                                         ('gain_raw', 'Generalization', '-', '#26364a'),
                                         ('steer_raw', 'Steering', '-', colors[j])]:
            y = np.clip(arrays[key], 0, 1).mean(0)[:, j]
            ax.plot(steps, smooth(y, window), style, color=color, lw=1.7, label=label)
        ax.set_title(f"Latent {ref['latents'][j]}  ·  p={freq[j]:.2%}", fontsize=10)
        ax.set_ylim(-.025, 1.025)
        ax.grid(axis='y', color='#e8ebef', linewidth=.6)
        ax.set_xlabel('Training steps')
    axs.flat[0].legend(fontsize=7, frameon=False, loc='upper left')
    fig.savefig(out / 'per_latent.png')
    fig.savefig(out / 'per_latent.pdf')
    plt.close(fig)
    # Raw curves retain failures below zero, including large off-target responses.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout='constrained')
    for ax, key in zip(axes, ('steer_raw', 'gain_raw')):
        for j, color in enumerate(colors):
            ax.plot(steps, arrays[key].mean(0)[:, j], color=color, lw=1.2)
        ax.axhline(0, color='grey', lw=.7)
        ax.set(title=key, xlabel='Training steps', ylabel='Unclipped fidelity')
    fig.savefig(out / 'raw_fidelity.png')
    plt.close(fig)
    rows = []
    for s, d in enumerate(data):
        for j, latent in enumerate(ref['latents']):
            tg, cg = crossing(steps, arrays['gain_raw'][s, :, j])
            ts, cs = crossing(steps, arrays['steer_raw'][s, :, j])
            rows.append({'seed': d['config']['seed'], 'latent': latent, 'frequency': freq[j],
                         't_gain_50': tg, 'gain_status': cg, 't_steer_50': ts, 'steer_status': cs,
                         'steer_minus_gain': ts-tg if ts is not None and tg is not None else None,
                         'final_gain': arrays['gain_raw'][s, -1, j],
                         'final_steer': arrays['steer_raw'][s, -1, j],
                         'final_patch': arrays['patch_raw'][s, -1, j]})
    with (out / 'timing.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    observed = [r for r in rows if r['steer_minus_gain'] is not None]
    if observed:
        fig, ax = plt.subplots(figsize=(4.6, 4.3), layout='constrained')
        for r in observed:
            ax.scatter(r['t_gain_50'], r['t_steer_50'], color=cmap(norm(r['frequency'])),
                       s=34, edgecolors='white', linewidths=.5)
        lo = min(min(r['t_gain_50'], r['t_steer_50']) for r in observed) * .85
        hi = max(max(r['t_gain_50'], r['t_steer_50']) for r in observed) * 1.15
        ax.plot([lo, hi], [lo, hi], '--', color='#929aa8', lw=1, label='Same emergence time')
        ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi),
               xlabel='Generalization: steps to 0.5', ylabel='Steering: steps to 0.5',
               title=f'{len(observed)}/{len(rows)} latent × seed pairs cross both thresholds')
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out / 'emergence_times.png')
        fig.savefig(out / 'emergence_times.pdf')
        plt.close(fig)
    (out / 'figure_notes.txt').write_text(
        f"Sources: {', '.join(str(p) for p in paths)}\n"
        f"Display smoothing: {window}-checkpoint centered moving average; raw dots retained.\n"
        "Fidelity = 1 - mean ||p - one_hot(counterfactual graph endpoint)||² / (1 - 1/num_classes).\n"
        "Zero: uniform prediction. One: perfect endpoint prediction. Same score for input change and hidden edit.\n"
        "Paired-effect fidelities are retained separately in JSON: gain_effect_raw / steer_effect_raw.\n"
        "Headline floors each seed's aggregate fidelity at zero before averaging; raw_fidelity.png retains negatives.\n"
        "Shading: +/-1.96 standard errors across seeds (absent for one seed); contexts are fixed across time.\n"
        "Timing: unsmoothed raw fidelity crosses 0.5 for three checkpoints; no own-ceiling rescaling.\n"
        "Paired contexts may share paths: per-pair SE in JSON is descriptive, not independent-seed uncertainty.\n"
        "Exact patch is a context-dependent control, not a steering vector. Coincident teacher endpoints remain in scoring.\n")
    print(f'Figures and raw timing table: {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('histories', nargs='+', type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--smooth-window', type=int, default=1)
    a = p.parse_args()
    plot(a.histories, a.out_dir, a.smooth_window)
