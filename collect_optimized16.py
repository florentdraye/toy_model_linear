"""Fetch optimized 16-latent histories and reproduce the archived plot style."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

from plot_emergence import plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=Path('runs/emergence_optimized16_summary'))
    a = p.parse_args()
    command = ('cd /fast/fdraye/toy_model_linear && tar --warning=no-file-changed -czf - '
               'emergence_optimized16_refined_s*/history.json')
    for _ in range(3):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 'fdraye@login.cluster.is.localnet', command], capture_output=True)
        if result.returncode == 0:
            break
    else:
        raise RuntimeError(result.stderr.decode())
    histories = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            data = json.loads(raw)
            seed = data['config']['seed']
            if seed not in range(46, 56):
                raise ValueError(f'unexpected seed {seed}')
            path = a.out_dir / f'seed_{seed}' / 'history.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            histories.append(path)
    histories.sort(key=lambda x: json.loads(x.read_text())['config']['seed'])
    status = {json.loads(x.read_text())['config']['seed']:
              {'step': json.loads(x.read_text())['history'][-1]['step'],
               'complete': json.loads(x.read_text())['complete']} for x in histories}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, indent=2))
    if histories:
        plot(histories, a.out_dir/'figures', window=3)
        runs = [json.loads(x.read_text()) for x in histories]
        shared = sorted(set.intersection(*[{r['step'] for r in run['history']} for run in runs]))
        coordinates = dict(seeds=[run['config']['seed'] for run in runs], steps=shared,
                           latents=runs[0]['latents'], frequency=runs[0]['frequency'])
        keys = ('gain_raw', 'steer_raw', 'patch_raw', 'random_raw', 'mean_raw', 'accuracy')
        arrays = {key: np.array([[{r['step']: r for r in run['history']}[step][key]
                                  for step in shared] for run in runs]) for key in keys}
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise ValueError('nonfinite curve data')
        np.savez_compressed(a.out_dir/'figures'/'curves.npz', **coordinates, **arrays)
        with (a.out_dir/'figures'/'curves.csv').open('w') as f:
            fields = ['seed', 'step', 'latent', 'frequency', *keys]
            writer = csv.DictWriter(f, fields); writer.writeheader()
            for si, seed in enumerate(coordinates['seeds']):
                for ti, step in enumerate(shared):
                    for li, latent in enumerate(coordinates['latents']):
                        writer.writerow(dict(seed=seed, step=step, latent=latent,
                            frequency=coordinates['frequency'][li],
                            **{key: arrays[key][si, ti, li] for key in keys}))
        summary = dict(complete=len(runs)==10 and all(run['complete'] for run in runs),
                       dimensions=['seed', 'checkpoint', 'latent'], **coordinates,
                       evaluation_bank_sha256=runs[0]['evaluation_bank_sha256'],
                       final={key: dict(mean=float(x[:, -1].mean()),
                                        latent_mean_min=float(x[:, -1].mean(0).min()),
                                        latent_mean_max=float(x[:, -1].mean(0).max()),
                                        individual_min=float(x[:, -1].min()))
                              for key, x in arrays.items()})
        (a.out_dir/'figures'/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
