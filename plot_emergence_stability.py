"""Compare raw steering and generalization of source and weight-averaged models."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def compare(source_path, averaged_path, out):
    source, averaged = [json.loads(Path(p).read_text()) for p in (source_path, averaged_path)]
    for key in ('latents', 'frequency', 'reference', 'model_config', 'evaluation_bank_sha256'):
        if source[key] != averaged[key]:
            raise ValueError(f'incompatible {key}')
    if source['config']['seed'] != averaged['config']['seed']:
        raise ValueError('need paired model seeds')
    if not averaged.get('model_transform', {}).get('decay'):
        raise ValueError('second input must be a model-weight EMA measurement')
    src = {r['step']: r for r in source['history']}
    rows = averaged['history']
    steps = np.array([r['step'] for r in rows])
    coverage = f'{len(steps)}/{len(src)} saved checkpoints'
    raw = [src[t] for t in steps]
    decay = averaged['model_transform']['decay']
    freq = np.array(source['frequency'])
    norm = LogNorm(freq.min(), freq.max())
    colors = plt.get_cmap('viridis_r')(norm(freq))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, layout='constrained')
    values = {}
    for i, (rr, label) in enumerate(((raw, 'Original model'), (rows, f'Model-weight EMA {decay}'))):
        for j, (key, metric) in enumerate((('steer_raw', 'Mean steering'), ('gain_raw', 'Generalization'))):
            a = np.array([r[key] for r in rr])
            values[f'{label}: {metric}'] = a
            ax = axes[i, j]
            for k, color in enumerate(colors):
                ax.plot(steps, a[:, k], color=color, lw=.9, alpha=.8)
            ax.set_title(f'{label} · {metric}')
            ax.grid(axis='y', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
        axes[i, 0].set_ylabel('Raw Brier skill')
    for ax in axes[-1]:
        ax.set_xlabel('Training steps')
    lower = min(a.min() for a in values.values())
    axes[0, 0].set_ylim(min(-.025, lower-.02), 1.025)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis_r'), ax=axes,
                 fraction=.018, label='Latent training frequency')
    fig.suptitle(f"Width {source['model_config']['d_model']} · seed {source['config']['seed']} · "
                 f"{len(freq)} latents · {coverage}\nFresh difference-of-means vectors · strength 1 · "
                 'fixed held-out examples · no curve smoothing', fontsize=12)
    fig.savefig(out / 'weight_ema_comparison.png', dpi=200)
    fig.savefig(out / 'weight_ema_comparison.pdf')
    plt.close(fig)
    mask = (steps[:-1] >= 10000) & (np.diff(steps) == source['config']['eval_every'])
    summary = {'source': str(source_path), 'averaged': str(averaged_path),
               'complete': averaged.get('complete', False), 'steps': steps.tolist(),
               'full_training_trajectory': sorted(steps.tolist()) == sorted(src),
               'adjacent_late_transitions': int(mask.sum()), 'metrics': {}}
    for label, a in values.items():
        summary['metrics'][label] = dict(final_mean=float(a[-1].mean()),
            final_min=float(a[-1].min()), late_mean=float(a[steps >= 10000].mean())
            if (steps >= 10000).any() else None,
            mean_absolute_change=float(np.abs(np.diff(a, axis=0))[mask].mean())
            if mask.any() else None)
    (out / 'weight_ema_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('averaged', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    compare(a.source, a.averaged, a.out_dir)
