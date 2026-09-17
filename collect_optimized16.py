"""Fetch optimized 16-latent histories and reproduce the archived plot style."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

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


if __name__ == '__main__':
    main()
