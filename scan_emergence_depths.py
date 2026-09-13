"""Diagnose last-token mean steering across every block of a saved final model.

Uses the final quarter of the original TRAINING calibration pairs, never test
scores. These paths also enter the full-support means: scores are calibration,
not held-out generalization. No direction optimization or strength search.
Run through submit_emergence_depths.sub on a GPU compute node.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess

import torch

from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import measure
from src.graph import Graph
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank
from train_emergence import atomic_json


def calibration_ids(saved, total_paths):
    train = saved['train'].sort().values
    if len(train.unique()) != len(train):
        raise ValueError('training support contains duplicates')
    allowed = torch.zeros(total_paths, dtype=torch.bool)
    allowed[train] = True
    pairs = saved['fit'][:, max(1, int(saved['fit'].shape[1] * .75)):]
    if pairs.shape[1] < 2 or not allowed[pairs].all():
        raise ValueError('need at least two training calibration pairs per latent')
    if allowed[saved['test']].any():
        raise ValueError('test paths overlap training support')
    return train, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--mean-batch', type=int, default=8192)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('submit this scan to a GPU compute node')
    if a.mean_batch < 1:
        raise ValueError('mean-batch must be positive')
    source = json.loads((a.run / 'history.json').read_text())
    if not source.get('complete'):
        raise ValueError('source training must be complete')
    step = source['history'][-1]['step']
    checkpoint = a.run / 'models' / f'step{step:06d}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    saved = torch.load(a.run / 'banks.pt', weights_only=True)
    train, pairs = calibration_ids(saved, len(paths['labels']))
    bank = {'off': paths['edge_seqs'][pairs[..., 0]].cuda(),
            'on': paths['edge_seqs'][pairs[..., 1]].cuda(),
            'y_off': paths['labels'][pairs[..., 0]].cuda(),
            'y_on': paths['labels'][pairs[..., 1]].cuda()}
    means_bank = UniformMeanBank(paths['edge_seqs'][train].cuda(),
                                paths['nodes'][train, source['config']['graph_layer']].cuda())
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    positions = [model.cfg.seq_len - 1]
    result = dict(complete=False, parent_run=str(a.run.resolve()), step=step,
                  seed=source['config']['seed'], latents=source['latents'],
                  reference=source['reference'], model_config=source['model_config'],
                  positions=positions, site='last', alpha=1.,
                  evaluation='final quarter of original training calibration pairs',
                  evaluation_overlaps_mean_support=True, test_scores_used=False,
                  pairs_per_latent=pairs.shape[1],
                  direction_estimator=means_bank.description(source['latents']),
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  records=[])
    a.out_dir.mkdir(parents=True, exist_ok=False)
    torch.save(pairs, a.out_dir / 'calibration_ids.pt')
    print(json.dumps({k: result[k] for k in ('host', 'gpu', 'step', 'pairs_per_latent')}), flush=True)
    for depth in range(1, len(model.blocks) + 1):
        vectors, means = means_bank.fit(model, source['latents'], depth, positions, a.mean_batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            metrics = measure(model, bank, vectors, depth, positions, source['config']['eval_batch'])
        record = dict(depth=depth, **metrics)
        record['patch_gain_max_difference'] = max(abs(x-y) for x, y in
                                                 zip(metrics['patch_raw'], metrics['gain_raw']))
        if depth == len(model.blocks) and record['patch_gain_max_difference'] > .002:
            raise ValueError('final-block exact patch does not reproduce target prediction')
        result['records'].append(record)
        torch.save(dict(vectors=vectors.cpu(), class_means=means.cpu()),
                   a.out_dir / f'depth{depth}.pt')
        result['complete'] = depth == len(model.blocks)
        atomic_json(a.out_dir / 'depth_scan.json', result)
        print(json.dumps({'depth': depth, **{k: sum(metrics[k])/len(vectors)
                          for k in ('gain_raw', 'steer_raw', 'patch_raw', 'random_raw')}}), flush=True)


if __name__ == '__main__':
    main()
