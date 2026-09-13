"""Refine saved constant steering vectors without retraining the model.

The probability-loss fit can flatten when the original prediction is confidently
wrong. Continue with log-loss gradients, retaining the best probability-loss
validation iterate (including the original vector). Test paths never select it.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import torch
from src.config import ModelConfig
from src.model import ToyTransformer
from src.graph import Graph
from src.data import enumerate_paths
from src.emergence import fit_directions, optimize_directions, measure
from train_emergence import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=400)
    a = p.parse_args()
    source = json.loads((a.run / 'history.json').read_text())
    if not source.get('complete'):
        raise ValueError('finish the source training run before refinement')
    if (a.out_dir / 'history.json').exists():
        raise FileExistsError('choose a fresh refinement output directory')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'directions').mkdir(exist_ok=True)
    c = source['config']
    result = copy.deepcopy(source)
    result['history'] = []
    result['complete'] = False
    result['parent_run'] = str(a.run.resolve())
    result['model_directory'] = str((a.run / 'models').resolve())
    result['refinement'] = {'objective': 'ce', 'steps': a.steps,
                            'initial': 'saved Brier vector', 'selection': 'validation Brier', 'radius_scale': 1.}
    result['refinement_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['refinement_host'] = socket.gethostname()
    result['refinement_gpu'] = torch.cuda.get_device_name()
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    ids = torch.load(a.run / 'banks.pt', weights_only=True)

    def bank(key):
        ix = ids[key]
        return {'off': paths['edge_seqs'][ix[..., 0]].cuda(),
                'on': paths['edge_seqs'][ix[..., 1]].cuda(),
                'y_off': paths['labels'][ix[..., 0]].cuda(),
                'y_on': paths['labels'][ix[..., 1]].cuda()}

    fit, test = bank('fit'), bank('test')
    shutil.copyfile(a.run / 'graph.pt', a.out_dir / 'graph.pt')
    shutil.copyfile(a.run / 'banks.pt', a.out_dir / 'banks.pt')
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    t0 = time.time()
    for old in source['history']:
        step = old['step']
        model.load_state_dict(torch.load(a.run / 'models' / f'step{step:06d}.pt', weights_only=True))
        initial = torch.load(a.run / 'directions' / f'step{step:06d}.pt', weights_only=True).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            means = fit_directions(model, fit, c['steer_depth'], result['positions'], c['eval_batch'])
            vectors, info = optimize_directions(model, fit, means, c['steer_depth'], result['positions'],
                                                 steps=a.steps, lr=c['direction_lr'], objective='ce', initial=initial)
            metrics = measure(model, test, vectors, c['steer_depth'], result['positions'], c['eval_batch'], means)
        row = {**old, **metrics, **info, 'brier_only_steer_raw': old['steer_raw']}
        result['history'].append(row)
        result['complete'] = step == source['history'][-1]['step']
        torch.save(vectors.cpu(), a.out_dir / 'directions' / f'step{step:06d}.pt')
        atomic_json(a.out_dir / 'history.json', result)
        print(f"step {step:5d} gain {sum(metrics['gain_raw'])/len(means):.3f} "
              f"steer {sum(metrics['steer_raw'])/len(means):.3f} ({time.time()-t0:.0f}s)", flush=True)
    from plot_emergence import plot
    plot([a.out_dir / 'history.json'], a.out_dir / 'figures', window=3)


if __name__ == '__main__':
    main()
