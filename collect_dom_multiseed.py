"""Copy atomically written histories from the cluster and regenerate local plots."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

from plot_dom_multiseed import plot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=Path('runs/dom_multiseed_summary'))
    p.add_argument('--first-seed', type=int, default=46)
    p.add_argument('--last-seed', type=int, default=95)
    a = p.parse_args()
    command = ('cd /fast/fdraye/toy_model_linear && '
               'tar --warning=no-file-changed -czf - dom_multiseed_s*/history.json')
    for attempt in range(3):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 'fdraye@login.cluster.is.localnet', command], capture_output=True)
        if result.returncode == 0:
            break
    else:
        raise RuntimeError(result.stderr.decode())
    paths = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            run = json.loads(raw)
            seed = run['config']['seed']
            if seed not in range(a.first_seed, a.last_seed + 1):
                raise ValueError(f'unexpected seed {seed}')
            path = a.out_dir / f'seed_{seed}' / 'history.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            paths.append(path)
    status = {json.loads(p.read_text())['config']['seed']:
              {'step': json.loads(p.read_text())['history'][-1]['step'],
               'complete': json.loads(p.read_text())['complete']} for p in paths}
    (a.out_dir/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, indent=2))
    expected = a.last_seed - a.first_seed + 1
    if len(paths) >= 2:
        plot(paths, a.out_dir/'figures', allow_partial=True)
    else:
        print(f'{len(paths)}/{expected} seeds have measurements; wait for at least two before plotting the mean.')


if __name__ == '__main__':
    main()
