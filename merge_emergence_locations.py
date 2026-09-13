"""Join disjoint, complete causal-EMA scan chunks without averaging any scores."""
import argparse
import json
from pathlib import Path
import shutil

def merge(paths, out):
    chunks = [json.loads((p / 'history.json').read_text()) for p in paths]
    ref = chunks[0]
    if not ref.get('model_transform', {}).get('decay'):
        raise ValueError('this merger requires causal model-weight EMA chunks')
    rows = {}
    owners = {}
    expected = sorted(set(range(0, ref['config']['steps']+1, ref['config']['eval_every'])) |
                      {ref['config']['steps']})
    for path, chunk in zip(paths, chunks):
        if not chunk.get('complete'):
            raise ValueError(f'unfinished chunk: {path}')
        for key in ('source', 'config', 'model_config', 'latents', 'frequency', 'cells',
                    'reference', 'selection', 'alphas', 'evaluation_bank_sha256', 'direction_estimator'):
            if chunk[key] != ref[key]:
                raise ValueError(f'incompatible {key}: {path}')
        transform = chunk['model_transform']
        for key in ('method', 'decay', 'initialization'):
            if transform[key] != ref['model_transform'][key]:
                raise ValueError(f'incompatible averaging {key}')
        last = max(r['step'] for r in chunk['history'])
        if transform['input_steps'] != [s for s in expected if s <= last]:
            raise ValueError('chunk did not average the complete causal snapshot prefix')
        for row in chunk['history']:
            step = row['step']
            if step in rows:
                raise ValueError(f'duplicate step {step}')
            if not (path / 'models' / f'step{step:06d}.pt').is_file():
                raise FileNotFoundError(f'averaged model missing at {path}, step {step}')
            if not (path / f'step{step:06d}.pt').is_file():
                raise FileNotFoundError(f'means missing at {path}, step {step}')
            rows[step], owners[step] = row, path
    if sorted(rows) != expected:
        raise ValueError('chunks do not cover the full requested trajectory')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    for step, path in owners.items():
        name = f'step{step:06d}.pt'
        (out / 'models' / name).symlink_to((path / 'models' / name).resolve())
        (out / name).symlink_to((path / name).resolve())
    shutil.copyfile(paths[0] / 'pair_ids.pt', out / 'pair_ids.pt')
    result = {**ref, 'history': [rows[s] for s in expected], 'complete': True,
              'model_directory': str((out / 'models').resolve()),
              'host': 'multiple compute nodes; see chunks',
              'chunks': [dict(path=str(p), host=c['host'], gpu=c['gpu'], git_commit=c['git_commit'],
                              steps=[r['step'] for r in c['history']]) for p, c in zip(paths, chunks)],
              'model_transform': {**ref['model_transform'], 'input_steps': expected}}
    # Keep this metadata-only merger free of NumPy/PyTorch imports on login nodes.
    tmp = out / 'history.tmp'
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(out / 'history.json')
    print(f'Merged {len(expected)} checkpoints into {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('chunks', type=Path, nargs='+')
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    merge(a.chunks, a.out_dir)
