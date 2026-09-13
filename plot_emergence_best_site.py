"""Plot held-out steering at calibration-selected single-token locations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_emergence import crossing, smooth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('histories', type=Path, nargs='+')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    data = sorted([json.loads(path.read_text()) for path in a.histories], key=lambda d: d['seed'])
    if not a.allow_partial and not all(d['complete'] for d in data):
        raise ValueError('incomplete histories; use --allow-partial for a progress plot')
    if len({d['seed'] for d in data}) != len(data):
        raise ValueError('duplicate seeds')
    for d in data[1:]:
        for key in ('latents', 'frequency', 'reference', 'graph_layer', 'model_config',
                    'alphas', 'selection', 'mean_estimator', 'reference_mean_estimator'):
            if d[key] != data[0][key]:
                raise ValueError(f'histories differ in {key}')
    steps = sorted(set.intersection(*[{r['step'] for r in d['history']} for d in data]))
    rows = [[next(r for r in d['history'] if r['step']==step) for step in steps] for d in data]
    gain = np.array([[r['gain_raw'] for r in seed] for seed in rows])
    groups = list(rows[0][0]['groups'])
    steering = {g: np.array([[r['groups'][g]['steer_raw'] for r in seed] for seed in rows]) for g in groups}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    title = 'Best single-token steering · selected on calibration, scored on held-out paths'
    if not all(d['complete'] for d in data):
        title += f' · partial through step {steps[-1]}'
    styles = [('unit', 'Best depth/token · strength 1', '#c75b39'),
              ('scaled', 'Best depth/token/strength', '#168b75')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, layout='constrained')
    for ax, method, label in zip(axes, ['rest', 'reference'], ['Target − rest means', 'Target − reference means']):
        ax.plot(steps, gain.mean((0, 2)), color='.25', ls='--', lw=2, label='Generalization')
        for mode, name, color in styles:
            values = steering[f'{method}_{mode}'].mean(-1)
            for seed_values in values:
                ax.plot(steps, seed_values, color=color, lw=.7, alpha=.22)
            ax.plot(steps, values.mean(0), color=color, lw=2, label=name)
        ax.axhline(0, color='.7', lw=.7)
        ax.set(title=label, xlabel='Training step', ylim=(-1.04, 1.04))
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('Raw target Brier skill')
    fig.suptitle(title, fontsize=12)
    fig.savefig(a.out_dir / 'best_site_steering.png', dpi=180)
    fig.savefig(a.out_dir / 'best_site_steering.pdf')
    plt.close(fig)
    # The user's original paired, frequency-colored curves, one condition at a time.
    colors = plt.get_cmap('viridis_r')(np.linspace(.15, .9, len(data[0]['latents'])))
    for group in groups:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True, sharey=True,
                                 layout='constrained')
        for ax, values, subtitle in zip(axes, [steering[group], gain],
                                         ['Steering at the selected location', 'Generalization']):
            average = values.mean(0)
            for j, color in enumerate(colors):
                ax.plot(steps, average[:, j], '.', color=color, ms=2, alpha=.3)
                ax.plot(steps, smooth(average[:, j], 3), color=color, lw=1.8,
                        label=f"{data[0]['latents'][j]} ({100*data[0]['frequency'][j]:.2f}%)")
            ax.axhline(0, color='.7', lw=.7)
            ax.axhline(.5, color='.75', lw=.7, ls=':')
            ax.set(title=subtitle, xlabel='Training step', ylim=(-1.04, 1.04))
        axes[0].set_ylabel('Raw target Brier skill')
        axes[1].legend(fontsize=7, ncol=2, loc='lower right', title='Latent (training frequency)')
        contrast, mode = group.split('_')
        mode_label = 'strength 1' if mode=='unit' else 'strength selected from 1, 2, 4, 8, 16'
        fig.suptitle(f'Target − {contrast} means · best depth/token per latent/checkpoint · {mode_label}', fontsize=11)
        fig.savefig(a.out_dir / f'{group}_steering_gain.png', dpi=180)
        fig.savefig(a.out_dir / f'{group}_steering_gain.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(len(data), 2, figsize=(12, 3*len(data)), sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    for si, d in enumerate(data):
        for mi, method in enumerate(['rest', 'reference']):
            ax = axes[si, mi]
            ax.plot(steps, gain[si].mean(-1), color='.25', ls='--', label='Generalization')
            for mode, name, color in styles:
                ax.plot(steps, steering[f'{method}_{mode}'][si].mean(-1), color=color, label=name)
            ax.axhline(0, color='.7', lw=.7)
            ax.set(title=f"Seed {d['seed']} · target − {method}", ylim=(-1.04, 1.04), ylabel='Raw Brier skill')
            if si==len(data)-1:
                ax.set_xlabel('Training step')
            if si==0:
                ax.legend(fontsize=8)
    fig.suptitle('Each model keeps its own best locations · all eight latents retained')
    fig.savefig(a.out_dir / 'per_seed.png', dpi=180)
    plt.close(fig)
    for field, label, max_value in [('position', 'Token index (0-based)', data[0]['model_config']['seq_len']-1),
                                    ('depth', 'Depth (0 = embedding)', data[0]['model_config']['n_blocks'])]:
        fig, axes = plt.subplots(len(data), 2, figsize=(12, 2.7*len(data)), sharex=True, squeeze=False,
                                 layout='constrained')
        for si, d in enumerate(data):
            for mi, method in enumerate(['rest', 'reference']):
                ax = axes[si, mi]
                values = np.array([[c[field] for c in r['groups'][f'{method}_scaled']['choices']] for r in rows[si]]).T
                im = ax.imshow(values, aspect='auto', origin='lower', interpolation='nearest',
                               cmap=plt.get_cmap('viridis', max_value+1), vmin=-.5, vmax=max_value+.5,
                               extent=[steps[0]-.5, steps[-1]+.5, -.5, len(d['latents'])-.5])
                ax.set(yticks=range(len(d['latents'])), yticklabels=d['latents'], ylabel='Latent',
                       title=f"Seed {d['seed']} · target − {method}")
                if si==len(data)-1:
                    ax.set_xlabel('Training step')
        fig.colorbar(im, ax=axes, ticks=range(max_value+1), label=label, shrink=.7)
        fig.suptitle(f'Calibration-selected {field} · strength-selected condition')
        fig.savefig(a.out_dir / f'selected_{field}s.png', dpi=160)
        plt.close(fig)
    table = []
    final = []
    for si, d in enumerate(data):
        for group in groups:
            for j, latent in enumerate(d['latents']):
                for row in rows[si]:
                    choice = row['groups'][group]['choices'][j]
                    item = dict(seed=d['seed'], step=row['step'], latent=latent, group=group,
                                **choice, heldout_skill=row['groups'][group]['steer_raw'][j],
                                gain=row['gain_raw'][j])
                    table.append(item)
                    if row['step']==steps[-1]:
                        final.append(item)
    for filename, values in [('all_choices.csv', table), ('final_choices.csv', final)]:
        with (a.out_dir / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    timing = []
    for si, d in enumerate(data):
        for group in groups:
            for j, latent in enumerate(d['latents']):
                tg, cg = crossing(steps, gain[si, :, j])
                ts, cs = crossing(steps, steering[group][si, :, j])
                timing.append(dict(seed=d['seed'], latent=latent, group=group,
                                   t_gain=tg, gain_status=cg, t_steer=ts, steer_status=cs))
    with (a.out_dir / 'timing.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    print(f"Saved {len(data)}-seed comparison through step {steps[-1]} to {a.out_dir}")


if __name__ == '__main__':
    main()
