"""Replay saved models with full-support difference-of-means edits.

No retraining, optimization, validation selection, or steering-strength sweep.
The original held-out pairs and generalization measurements are retained.
Run GPU work through submit_emergence_means.sub, never on a login node/laptop.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time

import torch

from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import measure
from src.graph import Graph
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank
from train_emergence import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='original training run with models/ and banks.pt')
    p.add_argument('--baseline', type=Path, help='optional refined history to preserve generalization from')
    p.add_argument('--evaluation-run', type=Path,
                   help='use this run’s fixed targets/pairs to compare model capacities; recompute generalization')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--mean-batch', type=int, default=8192)
    p.add_argument('--site', choices=['last', 'token', 'suffix', 'all'], default='last')
    p.add_argument('--contrast', choices=['rest', 'reference'], default='rest')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('submit this full-support replay to a GPU compute node')
    if a.mean_batch < 1:
        raise ValueError('mean-batch must be positive')
    original = json.loads((a.run / 'history.json').read_text())
    source = json.loads(a.baseline.read_text()) if a.baseline else original
    if a.baseline and a.evaluation_run:
        raise ValueError('a new evaluation bank cannot preserve an old baseline curve')
    evaluation = (json.loads((a.evaluation_run / 'history.json').read_text())
                  if a.evaluation_run else source)
    if a.evaluation_run:
        for key in ('graph_config', 'reference', 'metric'):
            if evaluation[key] != original[key]:
                raise ValueError(f'evaluation run differs in {key}')
        for key in ('data_seed', 'graph_layer', 'frequency_ratio', 'train_frac'):
            if evaluation['config'][key] != original['config'][key]:
                raise ValueError(f'evaluation run differs in {key}')
    if not source.get('complete') or not original.get('complete'):
        raise ValueError('source training and baseline must be complete')
    for key in ('model_config', 'graph_config', 'latents', 'reference'):
        if source[key] != original[key]:
            raise ValueError(f'baseline differs in {key}')
    for key in ('seed', 'data_seed', 'graph_layer', 'frequency_ratio'):
        if source['config'][key] != original['config'][key]:
            raise ValueError(f'baseline differs in {key}')
    if [r['step'] for r in source['history']] != [r['step'] for r in original['history']]:
        raise ValueError('baseline checkpoints differ')
    for row in source['history']:
        if not (a.run / 'models' / f"step{row['step']:06d}.pt").is_file():
            raise FileNotFoundError(f"missing model at step {row['step']}")
    # Protect previous runs, including partially written ones.
    a.out_dir.mkdir(parents=True, exist_ok=False)
    (a.out_dir / 'directions').mkdir()
    (a.out_dir / 'class_means').mkdir()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    c = source['config']
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    original_ids = torch.load(a.run / 'banks.pt', weights_only=True)
    ids = (torch.load(a.evaluation_run / 'banks.pt', weights_only=True)
           if a.evaluation_run else original_ids)
    for key in ('train', 'test_support'):
        if not torch.equal(ids[key].sort().values, original_ids[key].sort().values):
            raise ValueError(f'evaluation run has a different {key} support')
    train = ids['train'].sort().values
    if len(train.unique()) != len(train):
        raise ValueError('mean support must contain unique paths')
    allowed = torch.zeros(len(paths['labels']), dtype=torch.bool)
    allowed[train] = True
    if allowed[ids['test']].any():
        raise ValueError('evaluation paths overlap the mean support')
    means_bank = UniformMeanBank(paths['edge_seqs'][train].cuda(),
                                paths['nodes'][train, c['graph_layer']].cuda())
    ix = ids['test']
    if ix.shape[0] != len(evaluation['latents']):
        raise ValueError('evaluation bank and latent list differ')
    if not (paths['nodes'][ix[..., 0], c['graph_layer']] == evaluation['reference']).all():
        raise ValueError('evaluation off paths do not carry the reference')
    if not (paths['nodes'][ix[..., 1], c['graph_layer']] == torch.tensor(evaluation['latents'])[:, None]).all():
        raise ValueError('evaluation on paths do not carry the target')
    test = {'off': paths['edge_seqs'][ix[..., 0]].cuda(),
            'on': paths['edge_seqs'][ix[..., 1]].cuda(),
            'y_off': paths['labels'][ix[..., 0]].cuda(),
            'y_on': paths['labels'][ix[..., 1]].cuda()}
    length = source['model_config']['seq_len']
    positions = ([length - 1] if a.site == 'last' else [c['graph_layer'] - 1]
                 if a.site == 'token' else list(range(c['graph_layer'] - 1
                                                    if a.site == 'suffix' else 0, length)))
    reference = source['reference'] if a.contrast == 'reference' else None
    result = copy.deepcopy(source)
    if a.evaluation_run:
        for key in ('latents', 'frequency', 'fit_pairs_actual', 'eval_pairs_actual',
                    'tracked_frequency_ranks'):
            result[key] = evaluation[key]
        for key in ('num_latents', 'latent_ranks', 'fit_pairs', 'eval_pairs'):
            result['config'][key] = evaluation['config'][key]
        result['evaluation_run'] = str(a.evaluation_run.resolve())
    for key in list(result):
        if key.startswith('refinement'):
            del result[key]
    result.update(history=[], complete=False, positions=positions,
                  parent_run=str(a.run.resolve()),
                  baseline_history=str(a.baseline.resolve()) if a.baseline else str(a.run / 'history.json'),
                  model_directory=str((a.run / 'models').resolve()),
                  direction_estimator=means_bank.description(evaluation['latents'], reference),
                  evaluation_bank_sha256=hashlib.sha256(ix.numpy().tobytes()).hexdigest(),
                  training_support_sha256=hashlib.sha256(ids['train'].numpy().tobytes()).hexdigest(),
                  remeasurement_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  remeasurement_host=socket.gethostname(),
                  remeasurement_gpu=torch.cuda.get_device_name())
    result['config'].update(site=a.site, direction_method=('uniform-mean' if reference is None
                                                          else 'uniform-reference-mean'),
                            direction_steps=0, direction_lr=0., mean_batch=a.mean_batch)
    result['direction_estimator']['positions'] = positions
    shutil.copyfile(a.run / 'graph.pt', a.out_dir / 'graph.pt')
    torch.save(ids, a.out_dir / 'banks.pt')
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    result['num_parameters'] = model.num_params()
    print(json.dumps({k: result[k] for k in ('remeasurement_host', 'remeasurement_gpu',
                                           'positions', 'direction_estimator')}), flush=True)
    t0 = time.time()
    for old in source['history']:
        step = old['step']
        model.load_state_dict(torch.load(a.run / 'models' / f'step{step:06d}.pt', weights_only=True))
        vectors, class_means = means_bank.fit(model, evaluation['latents'], c['steer_depth'],
                                              positions, a.mean_batch, reference=reference)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            metrics = measure(model, test, vectors, c['steer_depth'], positions, c['eval_batch'])
        # The user's generalization curve is copied exactly, not refitted or
        # resampled. Check re-evaluation too, to catch the wrong model or bank.
        if a.evaluation_run:
            # Old per-latent arrays describe a different bank and must not leak
            # into the expanded measurement. Keep only training scalars.
            row = {k: old[k] for k in ('step', 'examples_seen', 'train_loss')}
            row.update(metrics)
        else:
            discrepancy = max(abs(x-y) for x, y in zip(metrics['gain_raw'], old['gain_raw']))
            if discrepancy > .002:
                raise ValueError(f'generalization replay mismatch at {step}: {discrepancy}')
            row = {k: v for k, v in old.items() if not k.startswith('direction_') and
                   k != 'brier_only_steer_raw'}
            row.update(metrics)
            for key in ('gain_raw', 'gain_se', 'gain_effect_raw', 'accuracy',
                        'reference_accuracy', 'effect_fraction'):
                row[key] = old[key]
            row['generalization_replay_max_difference'] = discrepancy
        result['history'].append(row)
        torch.save(vectors.cpu(), a.out_dir / 'directions' / f'step{step:06d}.pt')
        torch.save(class_means.cpu(), a.out_dir / 'class_means' / f'step{step:06d}.pt')
        result['complete'] = len(result['history']) == len(source['history'])
        atomic_json(a.out_dir / 'history.json', result)
        print(f"step {step:5d} gain {sum(row['gain_raw'])/len(vectors):.3f} "
              f"steer {sum(row['steer_raw'])/len(vectors):.3f} ({time.time()-t0:.0f}s)", flush=True)
    from plot_emergence import plot
    plot([a.out_dir / 'history.json'], a.out_dir / 'figures', window=3)


if __name__ == '__main__':
    main()
