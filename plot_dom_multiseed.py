"""Plot equal-weight model-seed means without temporal smoothing or clipping."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def load_runs(paths, allow_partial=False):
    runs = [json.loads(p.read_text()) for p in paths]
    runs.sort(key=lambda r: r['config']['seed'])
    seeds = [r['config']['seed'] for r in runs]
    if len(seeds) != len(set(seeds)) or len(seeds) < 2:
        raise ValueError('need distinct model seeds')
    for r in runs:
        for key in ('latents', 'frequency', 'reference', 'evaluation_bank_sha256',
                    'training_support_sha256', 'model_config'):
            if r[key] != runs[0][key]:
                raise ValueError(f'incompatible runs: {key}')
        for key in ('eval_every', 'steps', 'best_locations', 'graph_seed', 'data_seed'):
            if r['config'][key] != runs[0]['config'][key]:
                raise ValueError(f'incompatible protocol: {key}')
        if not allow_partial and not r.get('complete'):
            raise ValueError('unfinished run; use --allow-partial for explicitly provisional figures')
    common = sorted(set.intersection(*[{x['step'] for x in r['history']} for r in runs]))
    if not common:
        raise ValueError('no shared checkpoints')
    if not allow_partial and any([x['step'] for x in r['history']] != common for r in runs):
        raise ValueError('checkpoint grids differ')
    rows = [[{x['step']: x for x in r['history']}[t] for t in common] for r in runs]
    keys = ('gain_raw', 'steer_raw', 'patch_raw', 'accuracy', 'steering_accuracy', 'patch_accuracy')
    arrays = {k: np.array([[x[k] for x in rr] for rr in rows]) for k in keys}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('nonfinite scores')
    return runs, np.array(common), rows, arrays


def plot(paths, out, allow_partial=False):
    runs, steps, rows, data = load_runs(paths, allow_partial)
    out.mkdir(parents=True, exist_ok=True)
    n = len(runs)
    latents, frequency = runs[0]['latents'], np.array(runs[0]['frequency'])
    seeds = [r['config']['seed'] for r in runs]
    provisional = not all(r.get('complete') for r in runs)
    title = f'Width 512, six blocks — {n} model seeds, 32 latents'
    subtitle = 'Same 462 held-out pairs per latent; training-selected locations; strength 1; no smoothing or EMA'
    if provisional:
        title = 'PROVISIONAL through step ' + str(steps[-1]) + ' — ' + title
    colors = plt.get_cmap('viridis')(LogNorm(frequency.min(), frequency.max())(frequency))
    def save(fig, name):
        fig.savefig(out / (name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(out / (name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    def decorate(ax, ylabel='Held-out Brier gain (raw)'):
        ax.set(xlabel='Training step', ylabel=ylabel)
        ax.axhline(0, color='.6', lw=.5)
        ax.axhline(1, color='.6', lw=.5)
        ax.grid(alpha=.15)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, key, name in zip(axes, ('steer_raw', 'gain_raw'), ('DoM steering: seed means', 'Generalization: seed means')):
        for j, c in enumerate(colors):
            ax.plot(steps, data[key].mean(0)[:, j], color=c, lw=1.1)
        ax.set_title(name); decorate(ax)
    fig.suptitle(title+'\n'+subtitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, .94, .88))
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(frequency.min(), frequency.max()), cmap='viridis'),
                 ax=axes, label='Latent training probability', fraction=.025, pad=.03)
    save(fig, 'dom_vs_generalization')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # Paired seed bootstrap: resample whole runs, never independent checkpoints/latents.
    weights = np.random.default_rng(914).multinomial(n, np.ones(n)/n, size=4000)/n
    series = [('gain_raw', 'Generalization', '#2468b4'), ('steer_raw', 'DoM steering', '#df7d25'),
              ('patch_raw', 'Exact patch at DoM location', '#30965b')]
    for key, label, color in series:
        y = data[key].mean(2)
        lower, upper = np.quantile(weights @ y, [.025, .975], axis=0)
        axes[0].plot(steps, y.mean(0), color=color, label=label)
        axes[0].fill_between(steps, lower, upper, color=color, alpha=.18)
    axes[0].set_title('Mean across latents; 95% seed-bootstrap interval')
    axes[0].legend(fontsize=8)
    for i, seed in enumerate(seeds):
        axes[1].plot(steps, data['steer_raw'][i].mean(1), label=str(seed), lw=1)
    axes[1].set_title('DoM steering: every model seed')
    axes[1].legend(title='Seed', ncol=2, fontsize=7)
    gap = (data['gain_raw']-data['steer_raw']).mean(2)
    lo, hi = np.quantile(weights @ gap, [.025, .975], axis=0)
    axes[2].plot(steps, gap.mean(0), color='#7554a3')
    axes[2].fill_between(steps, lo, hi, color='#7554a3', alpha=.2)
    axes[2].set_title('Generalization minus steering; paired interval')
    for ax in axes: decorate(ax)
    axes[2].set_ylabel('Brier gain difference')
    fig.suptitle(title+'\nRaw averages; pointwise intervals resample whole model seeds', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .88)); save(fig, 'seed_summary')
    with PdfPages(out/'individual_seeds.pdf') as pdf:
        for start in range(0, n, 10):
            fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
            for i, ax in enumerate(axes.flat, start=start):
                if i >= min(start+10, n): ax.set_visible(False); continue
                for key, label, color in series:
                    ax.plot(steps, data[key][i].mean(1), label=label, color=color, lw=1)
                decorate(ax); ax.set_title(f'Seed {seeds[i]}')
            axes.flat[0].legend(fontsize=6)
            fig.suptitle(title+f'\nSeeds {start+1}–{min(start+10,n)}; each panel averages 32 latents', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .92)); pdf.savefig(fig)
            if start == 0: fig.savefig(out/'individual_seeds.png', dpi=170, bbox_inches='tight')
            plt.close(fig)
    with PdfPages(out/'all_latents.pdf') as pdf:
        for start in range(0, len(latents), 8):
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
            for j, ax in zip(range(start, min(start+8, len(latents))), axes.flat):
                for key, label, color in series[:2]:
                    y = data[key][:, :, j]
                    ax.plot(steps, y.mean(0), color=color, label=label)
                    ax.fill_between(steps, y.mean(0)-y.std(0, ddof=1),
                                    y.mean(0)+y.std(0, ddof=1), color=color, alpha=.18)
                ax.set_title(f'Latent {latents[j]} — p={frequency[j]:.4f}')
                decorate(ax)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(title+'\nBands: ±1 standard deviation across seeds (not uncertainty of the mean)', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .91)); pdf.savefig(fig)
            if start == 0: fig.savefig(out/'latent_examples.png', dpi=160, bbox_inches='tight')
            plt.close(fig)
    columns = ['seed', 'step', 'latent', 'frequency', *data.keys(), 'depth', 'positions', 'train_loss']
    with (out/'curves.csv').open('w') as f:
        writer = csv.DictWriter(f, columns); writer.writeheader()
        for i, seed in enumerate(seeds):
            for t, step in enumerate(steps):
                for j, latent in enumerate(latents):
                    cell = rows[i][t]['choices'][j]
                    writer.writerow(dict(seed=seed, step=step, latent=latent, frequency=frequency[j],
                        **{key: a[i, t, j] for key, a in data.items()}, depth=cell['depth'],
                        positions=' '.join(map(str, cell['positions'])), train_loss=rows[i][t]['train_loss']))
    np.savez_compressed(out/'curves.npz', steps=steps, seeds=seeds, latents=latents, frequency=frequency, **data)
    summary = dict(complete=not provisional, seeds=seeds, checkpoints=len(steps), last_step=int(steps[-1]),
                   bootstrap='4000 whole-seed resamples; pointwise percentile 95% intervals; rng 914',
                   array_dimensions=['seed', 'checkpoint', 'latent'],
                   evaluation_bank_sha256=runs[0]['evaluation_bank_sha256'],
                   final={key: dict(mean=float(a[:, -1].mean()),
                                   per_seed=a[:, -1].mean(1).tolist()) for key, a in data.items()})
    # Late-time variation is descriptive: it includes real parameter motion.
    late = steps >= 12000
    if late.sum() >= 2:
        summary['late_step_change_rms'] = {
            key: dict(seed_average=float(np.sqrt(np.mean(np.diff(a.mean(0)[late], axis=0)**2))),
                      individual_runs=float(np.sqrt(np.mean(np.diff(a[:, late], axis=1)**2))))
            for key, a in data.items() if key in ('gain_raw', 'steer_raw')}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('histories', type=Path, nargs='+')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    plot(args.histories, args.out_dir, args.allow_partial)
