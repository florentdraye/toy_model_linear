"""Check a larger vector-fitting budget at observed transition checkpoints.

GPU-only diagnostic. Select checkpoints by generalization's first 0.5 crossing,
refit with 3x iterations, select vectors on calibration validation, and report
test differences without changing the already saved headline measurements.
"""
import argparse
import json
from pathlib import Path
import torch
from src.config import ModelConfig
from src.model import ToyTransformer
from src.graph import Graph
from src.data import enumerate_paths
from src.emergence import fit_directions, optimize_directions, measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--multiplier', type=int, default=3)
    p.add_argument('--objective', choices=['brier', 'ce', 'hybrid'], default='brier')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    result = json.loads((a.run / 'history.json').read_text())
    c = result['config']
    selected = {}
    for j, latent in enumerate(result['latents']):
        row = next((h for h in result['history'] if h['gain_raw'][j] >= .5), None)
        if row:
            selected.setdefault(row['step'], []).append(j)
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    ids = torch.load(a.run / 'banks.pt', weights_only=True)

    def bank(key):
        ix = ids[key]
        return {'off': paths['edge_seqs'][ix[..., 0]].cuda(),
                'on': paths['edge_seqs'][ix[..., 1]].cuda(),
                'y_off': paths['labels'][ix[..., 0]].cuda(),
                'y_on': paths['labels'][ix[..., 1]].cuda()}

    fit, test = bank('fit'), bank('test')
    model = ToyTransformer(ModelConfig(**result['model_config'])).cuda().eval()
    output = []
    for step, latent_indices in sorted(selected.items()):
        model.load_state_dict(torch.load(a.run / 'models' / f'step{step:06d}.pt', weights_only=True))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            means = fit_directions(model, fit, c['steer_depth'], result['positions'], c['eval_batch'])
            initial = (torch.load(a.run / 'directions' / f'step{step:06d}.pt', weights_only=True).cuda()
                       if a.objective != 'brier' else None)
            vectors, info = optimize_directions(model, fit, means, c['steer_depth'], result['positions'],
                                                 steps=c['direction_steps']*a.multiplier, lr=c['direction_lr'],
                                                 objective=a.objective, initial=initial)
            metrics = measure(model, test, vectors, c['steer_depth'], result['positions'], c['eval_batch'], means)
        old = next(h for h in result['history'] if h['step'] == step)
        row = {'step': step, 'crossing_latents': [result['latents'][j] for j in latent_indices],
               'original_steer': old['steer_raw'], 'refined_steer': metrics['steer_raw'],
               'gain': metrics['gain_raw'], 'original_iterations': c['direction_steps'],
               'refined_iterations': c['direction_steps']*a.multiplier, 'objective': a.objective, **info}
        output.append(row)
        print(json.dumps(row), flush=True)
        (a.run / f'solver_audit_{a.objective}.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
