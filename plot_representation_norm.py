"""Plot unsmoothed hidden-state norms; no model execution."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    data = json.loads((a.run / 'history.json').read_text())
    if not data['complete']:
        raise ValueError('Measurement is not complete')
    steps = np.array([r['step'] for r in data['history']])
    norms = np.array([r['mean_norm'] for r in data['history']])
    assert np.isfinite(norms).all() and (norms >= 0).all()
    assert len(np.unique(steps)) == len(steps) and (np.diff(steps) > 0).all()
    out = a.run / 'figures'
    out.mkdir(exist_ok=True)
    freq = np.array(data['frequency'])
    color_norm = LogNorm(freq.min(), freq.max())
    cmap = plt.get_cmap('viridis')
    with (out / 'norms.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'latent', 'depth', 'token_index', 'mean_l2_norm', 'std_l2_norm'])
        for t, row in enumerate(data['history']):
            for k, latent in enumerate(data['latents']):
                for d in range(norms.shape[2]):
                    for pos in range(norms.shape[3]):
                        writer.writerow([int(steps[t]), latent, d, pos, norms[t, k, d, pos], row['std_norm'][k][d][pos]])
    for pos in range(norms.shape[3]):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout='constrained')
        for depth, ax in enumerate(axes.flat, 1):
            for k in range(len(freq)):
                ax.plot(steps, norms[:, k, depth, pos], color=cmap(color_norm(freq[k])), lw=1, alpha=.8)
            ax.set_title(f'After transformer block {depth}')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.set_xlabel('Training step')
            ax.set_ylabel(r'Mean $\|h\|_2$')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        fig.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), ax=axes, label='Latent training probability', shrink=.8)
        fig.suptitle(f'Representation norm — original model (no EMA)\nToken {pos + 1} (index {pos}); 32 latents, {data["examples_per_latent"]} fixed held-out examples each; width 512, seed 46', fontsize=13)
        name = 'representation_norm' if pos == data['graph_layer'] - 1 else f'representation_norm_token{pos}'
        fig.savefig(out / f'{name}.png', dpi=170)
        fig.savefig(out / f'{name}.pdf')
        plt.close(fig)
    print(out / 'representation_norm.png')


if __name__ == '__main__':
    main()
