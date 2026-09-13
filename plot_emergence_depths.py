"""Plot completed last-token depth scans; retain raw negative calibration scores."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scans', type=Path, nargs='+')
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    scans = [json.loads(path.read_text()) for path in a.scans]
    first = scans[0]
    depths = [r['depth'] for r in first['records']]
    for scan in scans:
        if not scan['complete'] or [r['depth'] for r in scan['records']] != depths:
            raise ValueError('scans must be complete with matching depths')
        for key in ('latents', 'reference', 'step', 'positions', 'alpha', 'model_config',
                    'evaluation', 'direction_estimator'):
            if scan[key] != first[key]:
                raise ValueError(f'scans differ in {key}')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    columns = {'depth': depths}
    for key, label, color, style in (
            ('gain_raw', 'Clean target', '0.4', '--'),
            ('steer_raw', 'Difference of means', '#d55e00', '-'),
            ('patch_raw', 'Exact last-token patch', '#0072b2', '-'),
            ('random_raw', 'Norm-matched random', '#009e73', ':')):
        values = np.array([[np.mean(row[key]) for row in scan['records']] for scan in scans])
        average = values.mean(0)
        columns[key] = average
        for seed_values in values:
            ax.plot(depths, seed_values, linestyle=style, color=color, alpha=.2, lw=1)
        ax.plot(depths, average, linestyle=style, color=color, marker='o', label=label, lw=2)
    ax.axhline(0, color='0.8', lw=1)
    ax.set(xticks=depths, xlabel='Post-block intervention depth',
           ylabel='Raw target Brier skill', ylim=(-1.04, 1.06),
           title=f"Last-token steering at step {first['step']:,}\n"
                 f"Training calibration pairs · {len(scans)} seeds · strength 1")
    ax.legend(loc='center left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(a.out_dir / 'depth_scan.png', dpi=180)
    fig.savefig(a.out_dir / 'depth_scan.pdf')
    plt.close(fig)
    with (a.out_dir / 'depth_scan.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


if __name__ == '__main__':
    main()
