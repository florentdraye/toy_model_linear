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
    a = p.parse_args()
    command = ('cd /fast/fdraye/toy_model_linear && '
               'tar -czf - dom_multiseed_s*/history.json')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'fdraye@login.cluster.is.localnet', command], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    paths = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            run = json.loads(raw)
            seed = run['config']['seed']
            if seed not in range(46, 56):
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
    if len(paths) == 10:
        plot(paths, a.out_dir/'figures', allow_partial=True)
    else:
        print(f'{len(paths)}/10 seeds have measurements; wait for all seeds before plotting the mean.')


if __name__ == '__main__':
    main()
