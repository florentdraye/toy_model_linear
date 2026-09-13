"""Plot token-location and mean-definition audits, preserving raw failures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audits', nargs='+', type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    a = p.parse_args()
    data = [json.loads(path.read_text()) for path in a.audits]
    if not all(d['complete'] for d in data):
        raise ValueError('all audits must be complete')
    for d in data[1:]:
        for key in ('latents', 'reference', 'sites', 'step', 'mean_support', 'fixed_test_protocol'):
            if d[key] != data[0][key]:
                raise ValueError(f'audits differ in {key}')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    length = len(data[0]['sites']['all'])
    depths = sorted({r['depth'] for r in data[0]['records']})

    def value(d, depth, site, method, alpha=1.):
        return next(r['mean_skill'] for r in d['records'] if
                    (r['depth'], r['site'], r['method'], r['alpha']) == (depth, site, method, alpha))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.1]})
    matrix = np.array([[np.mean([value(d, depth, f'token{j}', 'patch') for d in data])
                        for j in range(length)] for depth in depths])
    im = axes[0].imshow(matrix, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                         color='white' if abs(val)>.65 else 'black', fontsize=9)
    axes[0].set(xticks=range(length), xticklabels=[str(i) for i in range(length-1)]+['5\n(last)'],
                yticks=range(len(depths)), yticklabels=depths, xlabel='Edited token index',
                ylabel='Post-block depth', title='Where exact patches work\nTraining calibration pairs')
    fig.colorbar(im, ax=axes[0], shrink=.8, label='Raw target Brier skill')
    labels = ['Last token\n(target − rest)', 'Last token\n(target − reference)',
              'Tokens 2–5\n(target − rest)', 'Tokens 2–5\n(target − reference)']
    settings = [('token5', 'dom'), ('token5', 'reference'), ('suffix', 'dom'), ('suffix', 'reference')]
    vals = np.array([[next(r['mean_skill'] for r in d['test_records'] if
                          (r['site'], r['method']) == s) for s in settings] for d in data])
    colors = ['#c05a3d', '#c05a3d', '#c05a3d', '#248c73']
    axes[1].bar(range(4), vals.mean(0), color=colors, width=.65, alpha=.85)
    for i, mean in enumerate(vals.mean(0)):
        axes[1].text(i, mean + (.04 if mean>=0 else -.04), f'{mean:.3f}',
                     ha='center', va='bottom' if mean>=0 else 'top', fontweight='bold')
        axes[1].scatter(np.full(len(data), i), vals[:, i], color='black', s=14, zorder=3)
    axes[1].axhline(0, color='.5', lw=.8)
    axes[1].set(xticks=range(4), xticklabels=labels, ylim=(-1.08, 1.12), ylabel='Raw target Brier skill',
                title='Both changes matter\nHeld-out pairs · block 1 · strength 1')
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Analytic means, no optimization · final checkpoints · {len(data)} seeds', fontsize=13)
    fig.savefig(a.out_dir / 'intervention_diagnosis.png', dpi=180)
    fig.savefig(a.out_dir / 'intervention_diagnosis.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True, layout='constrained')
    alphas = [1., 2., 4., 8., 16.]
    for ax, method, title in zip(axes, ['dom', 'reference', 'matched'],
                                  ['Target − rest', 'Target − reference', 'Matched-pair mean']):
        for depth in depths:
            y = [np.mean([value(d, depth, 'token5', method, alpha) for d in data]) for alpha in alphas]
            ax.plot(alphas, y, '.-', label=f'Block {depth}')
        ax.set(xscale='log', xticks=alphas, xticklabels=['1', '2', '4', '8', '16'],
               xlabel='Insertion strength', title=title, ylim=(-1.05, 1.05))
        ax.axhline(0, color='.7', lw=.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Raw target Brier skill')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Last token only · FP32 · training calibration pairs')
    fig.savefig(a.out_dir / 'last_token_strength.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
