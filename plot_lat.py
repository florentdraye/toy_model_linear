"""LAT steering curves and reference-free mean absolute inter-class cosine."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


def absolute_cosine(vectors):
    v = vectors.double()
    norms = v.norm(dim=-1)
    if (norms == 0).any(): raise ValueError('zero direction in geometry')
    u = v / norms[..., None]
    cosine = u @ u.transpose(-1, -2)
    i, j = torch.triu_indices(v.shape[-2], v.shape[-2], 1)
    return cosine[..., i, j].abs().mean(-1).numpy(), cosine.numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--means-run', type=Path, required=True)
    p.add_argument('--expected-checkpoints', type=int, default=21)
    a = p.parse_args()
    torch.set_num_threads(2)
    files = list(a.run.glob('step*/history.json'))
    data = sorted([(json.loads(f.read_text()), f.parent) for f in files], key=lambda item: item[0]['step'])
    if len(data) != a.expected_checkpoints or not all(d['complete'] for d, _ in data):
        raise ValueError(f'Need {a.expected_checkpoints} completed checkpoints; found {len(data)}')
    rows, dirs = zip(*data)
    ref = rows[0]
    for r in rows:
        for key in ['latents', 'classes', 'frequency', 'bank_sha256', 'evaluation_bank_sha256', 'estimator', 'direction_scale', 'alphas']:
            assert r[key] == ref[key]
    steps = np.array([r['step'] for r in rows])
    assert len(np.unique(steps)) == len(steps)
    arrays = {key: np.array([r[key] for r in rows]) for key in ['steer_raw', 'steering_accuracy', 'gain_raw', 'accuracy',
              'patch_raw', 'random_raw', 'original_dom_steer', 'matched_mean_at_lat_site', 'full_dom_at_lat_site']}
    assert all(np.isfinite(value).all() for value in arrays.values())
    out = a.run / 'figures'; out.mkdir(exist_ok=True)
    with (out / 'steering_curves.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'latent', *arrays, 'selected_depth', 'selected_positions', 'selected_alpha'])
        for t, row in enumerate(rows):
            for k, latent in enumerate(ref['latents']):
                c = row['choices'][k]
                writer.writerow([row['step'], latent, *[arrays[key][t, k] for key in arrays], c['depth'], ' '.join(map(str, c['positions'])), c['alpha']])
    freq = np.array(ref['frequency']); norm = LogNorm(freq.min(), freq.max()); cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, layout='constrained')
    for ax, key, title in zip(axes, ['original_dom_steer', 'steer_raw', 'gain_raw'],
                              ['Previous DoM steering', 'LAT steering', 'Generalization']):
        for k in range(len(freq)):
            ax.plot(steps, arrays[key][:, k], color=cmap(norm(freq[k])), lw=1.2, alpha=.85)
        ax.set_title(title); ax.set_xlabel('Training step'); ax.grid(alpha=.2)
    minimum = min(float(arrays[key].min()) for key in ['original_dom_steer', 'steer_raw', 'gain_raw'])
    axes[0].set_ylim(min(-.025, minimum - .025), 1.025)
    axes[0].set_ylabel('Held-out Brier gain (raw, unclipped)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label='Latent training probability', shrink=.8)
    fig.suptitle(f'Original width-512 model — 32 target latents, {len(rows)} checkpoints\nSame fixed 462 held-out pairs per target; calibration-selected locations; no EMA or smoothing', fontsize=13)
    for ext in ['png', 'pdf']: fig.savefig(out / f'lat_steering.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True, layout='constrained')
    for ax, key, title in zip(axes, ['steering_accuracy', 'accuracy'], ['LAT steering accuracy', 'Unedited target accuracy']):
        for k in range(len(freq)): ax.plot(steps, arrays[key][:, k], color=cmap(norm(freq[k])), lw=1.1, alpha=.8)
        ax.set_title(title); ax.set_xlabel('Training step'); ax.set_ylim(-.025, 1.025); ax.grid(alpha=.2)
    axes[0].set_ylabel('Fraction of correct endpoint predictions')
    fig.suptitle('Actual classification accuracy on the fixed held-out pairs')
    for ext in ['png', 'pdf']: fig.savefig(out / f'lat_accuracy.{ext}', dpi=170)
    plt.close(fig)
    geometry, matrices = {key: [] for key in ['LAT', 'paired_mean', 'full_support_DoM']}, {key: [] for key in ['LAT', 'paired_mean', 'full_support_DoM']}
    energy = []
    pos = ref['config']['graph_layer'] - 1
    for row, directory in data:
        d = torch.load(directory / 'directions.pt', weights_only=True, map_location='cpu')
        means = torch.load(a.means_run / f"step{row['step']:06d}.pt", weights_only=True, map_location='cpu')['class_means']
        centered = means - means.mean(1, keepdim=True)
        for key, vectors in [('LAT', d['axes'][1:, :, pos]), ('paired_mean', d['paired_mean'][1:, :, pos]),
                             ('full_support_DoM', centered[1:, :, pos])]:
            value, matrix = absolute_cosine(vectors)
            geometry[key].append(value); matrices[key].append(matrix)
        energy.append(d['eigen_energy'][1:, :, pos].numpy())
    geometry = {key: np.array(value) for key, value in geometry.items()}
    np.savez_compressed(out / 'geometry.npz', steps=steps, classes=ref['classes'],
                        **{key + '_cosine': np.array(value) for key, value in matrices.items()}, eigen_energy=np.array(energy))
    with (out / 'geometry.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['step', 'block', 'method', 'mean_absolute_pairwise_cosine'])
        for t, step in enumerate(steps):
            for block in range(6):
                for key, values in geometry.items(): writer.writerow([step, block + 1, key, values[t, block]])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout='constrained')
    for block, ax in enumerate(axes.flat):
        for key, label, color in [('LAT', 'LAT principal axes', '#1671b8'), ('paired_mean', 'Mean of the same contrast pairs', '#dd8b22'),
                                  ('full_support_DoM', 'Full-support DoM (global mean removed)', '#525f69')]:
            ax.plot(steps, geometry[key][:, block], color=color, lw=1.8, label=label)
        ax.set_title(f'Block {block + 1}'); ax.set_xlabel('Training step'); ax.set_ylabel('Mean absolute cosine across latent pairs')
        ax.set_ylim(0, 1); ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('LAT versus mean directions — all 100 latent classes, third token\nAverage |cosine| over 4,950 distinct pairs; no reference subtraction, EMA, or smoothing', fontsize=13)
    for ext in ['png', 'pdf']: fig.savefig(out / f'lat_geometry.{ext}', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    for key, label in [('original_dom_steer', 'Original DoM, own best locations'), ('steer_raw', 'LAT, own best locations'),
                       ('full_dom_at_lat_site', 'Full-support DoM at LAT locations, strength 1'),
                       ('matched_mean_at_lat_site', 'Matched-bag mean at LAT locations, strength 1'),
                       ('patch_raw', 'Exact patch at LAT locations')]:
        ax.plot(steps, arrays[key].mean(1), label=label)
    ax.set_xlabel('Training step'); ax.set_ylabel('Mean raw Brier gain across 32 targets'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ext in ['png', 'pdf']: fig.savefig(out / f'lat_controls.{ext}', dpi=170)
    plt.close(fig)
    summary = dict(complete=True, steps=steps.tolist(), checkpoint_count=len(rows),
                   final_mean_lat_gain=float(arrays['steer_raw'][-1].mean()), final_min_lat_gain=float(arrays['steer_raw'][-1].min()),
                   final_mean_dom_gain=float(arrays['original_dom_steer'][-1].mean()),
                   final_mean_steering_accuracy=float(arrays['steering_accuracy'][-1].mean()),
                   final_min_steering_accuracy=float(arrays['steering_accuracy'][-1].min()),
                   final_perfect_accuracy_latents=int((arrays['steering_accuracy'][-1] == 1).sum()),
                   final_geometry={key: value[-1].tolist() for key, value in geometry.items()},
                   max_eigen_residual=max(c['max_residual'] for r in rows for c in r['eigen_checks']),
                   max_exact_eigen_check_error=max(c['exact_check_error'] for r in rows for c in r['eigen_checks']),
                   max_generalization_replay_error=max(r['generalization_replay_max_difference'] for r in rows),
                   hosts=sorted(set(r['host'] for r in rows)), gpus=sorted(set(r['gpu'] for r in rows)))
    (a.run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
