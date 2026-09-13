"""Lightweight login-node monitor: release held scans after successful training.

Only reads JSON/Condor metadata and releases already-submitted jobs. It performs
no model computation. Job ids are explicit; no new jobs are submitted here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/fast/fdraye/toy_model_linear'))
    p.add_argument('--large-cluster', type=int, required=True)
    p.add_argument('--medium-cluster', type=int, required=True)
    p.add_argument('--scan-cluster', type=int, required=True)
    a = p.parse_args()
    released, previous = set(), {}
    while True:
        finished = 0
        for size, cluster, offset in [('large', a.large_cluster, 0), ('medium', a.medium_cluster, 3)]:
            history = subprocess.check_output(
                ['condor_history', str(cluster), '-limit', '3', '-autoformat', 'ProcId', 'JobStatus', 'ExitCode'],
                text=True)
            succeeded = {int(fields[0]) for line in history.splitlines()
                         if len(fields := line.split()) == 3 and fields[1:] == ['4', '0']}
            for proc, seed in enumerate((46, 47, 48)):
                name = f'{size}_s{seed}'
                train_path = a.root / f'emergence_{size}_means_s{seed}' / 'history.json'
                scan_path = a.root / f'emergence_{size}_locations_s{seed}' / 'history.json'
                train = json.loads(train_path.read_text()) if train_path.exists() else {}
                scan = json.loads(scan_path.read_text()) if scan_path.exists() else {}
                state = (train.get('history', [{}])[-1].get('step'),
                         scan.get('history', [{}])[-1].get('step'), scan.get('complete', False))
                if previous.get(name) != state:
                    print(json.dumps(dict(run=name, train_step=state[0], scan_step=state[1], scan_complete=state[2])), flush=True)
                    previous[name] = state
                job = f'{a.scan_cluster}.{offset+proc}'
                if proc in succeeded and train.get('complete') and job not in released and not scan:
                    subprocess.run(['condor_release', job], check=True)
                    released.add(job)
                    print(f'Released {job}: {name} training exited successfully.', flush=True)
                finished += bool(scan.get('complete'))
        if finished == 6:
            print('All six training runs and all six full location scans are complete.', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
