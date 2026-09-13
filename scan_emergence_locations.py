"""Select analytic reference-mean locations on training pairs, then score held-out.

Includes embeddings, every block, every single token, the graph-latent suffix,
and all tokens. Strength stays one. Explicit snapshots may be scanned while
training continues; saved model files must already exist.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import torch

from audit_emergence_interventions import capture, score_edit
from remeasure_emergence_best_site import all_depth_means
from scan_emergence_depths import calibration_ids
from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import measure
from src.graph import Graph
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank
from train_emergence import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('submit this scan to a GPU compute node')
    source = json.loads((a.run / 'history.json').read_text())
    if not source.get('complete') and a.steps is None:
        raise ValueError('unfinished training requires explicit saved steps')
    rows = [r for r in source['history'] if a.steps is None or r['step'] in a.steps]
    if not rows or (a.steps is not None and set(a.steps) != {r['step'] for r in rows}):
        raise ValueError('requested checkpoints unavailable')
    for r in rows:
        if not (a.run / 'models' / f"step{r['step']:06d}.pt").is_file():
            raise FileNotFoundError(f"model snapshot at {r['step']} is unavailable")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    saved = torch.load(a.run / 'banks.pt', weights_only=True)
    train, pairs = calibration_ids(saved, len(paths['labels']))
    c = source['config']
    bank = UniformMeanBank(paths['edge_seqs'][train].cuda(),
                          paths['nodes'][train, c['graph_layer']].cuda())
    target_ix = [bank.classes.index(k) for k in source['latents']]
    ref_ix = bank.classes.index(source['reference'])
    test_ids = saved['test']
    test = dict(off=paths['edge_seqs'][test_ids[..., 0]].cuda(),
                on=paths['edge_seqs'][test_ids[..., 1]].cuda(),
                y_off=paths['labels'][test_ids[..., 0]].cuda(),
                y_on=paths['labels'][test_ids[..., 1]].cuda())
    labels = paths['labels'][pairs[..., 1]].cuda()
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    sites = {f'token {j}': [j] for j in range(model.cfg.seq_len)}
    sites['suffix'] = list(range(c['graph_layer'] - 1, model.cfg.seq_len))
    sites['all tokens'] = list(range(model.cfg.seq_len))
    cells = [dict(depth=d, site=name, positions=pos)
             for d in range(len(model.blocks)+1) for name, pos in sites.items()]
    result = dict(source=str(a.run), config=c, model_config=source['model_config'],
                  latents=source['latents'], frequency=source['frequency'],
                  cells=cells, reference=source['reference'],
                  selection='per latent and checkpoint, highest raw TRAINING calibration score; alpha 1',
                  calibration='fixed final quarter of fit pairs; overlaps training mean support',
                  evaluation='fixed held-out pairs; excluded from means and location selection',
                  evaluation_bank_sha256=hashlib.sha256(test_ids.numpy().tobytes()).hexdigest(),
                  direction_estimator=bank.description(source['latents'], source['reference']),
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  complete=False, history=[])
    torch.save(dict(calibration=pairs, test=test_ids), a.out_dir / 'pair_ids.pt')
    print(json.dumps({k: result[k] for k in ('host', 'gpu', 'selection')}), flush=True)
    start = time.time()
    for old in rows:
        step = old['step']
        model.load_state_dict(torch.load(a.run / 'models' / f'step{step:06d}.pt',
                                        map_location='cpu', weights_only=True))
        means = all_depth_means(model, bank)
        vectors = (means[:, target_ix] - means[:, ref_ix:ref_ix+1]).float()
        scores = []
        for depth in range(len(model.blocks)+1):
            hidden = capture(model, paths, pairs[..., 0], depth)
            for positions in sites.values():
                scores.append(score_edit(model, hidden, None, labels, depth,
                                         positions, vectors[depth])['skill'])
        scores = torch.tensor(scores, dtype=torch.float64)
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite calibration score')
        # Freeze the site choices before inspecting any held-out responses.
        chosen = scores.argmax(0).tolist()
        choices = [{**cells[i], 'calibration_skill': float(scores[i, j])}
                   for j, i in enumerate(chosen)]
        metrics = {}
        selected_vectors = []
        for j, cell in enumerate(choices):
            depth, pos = cell['depth'], cell['positions']
            v = vectors[depth, j:j+1, pos]
            selected_vectors.append(v.cpu())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                m = measure(model, {k: x[j:j+1] for k, x in test.items()}, v, depth, pos, c['eval_batch'])
            for key, value in m.items():
                metrics.setdefault(key, []).extend(value)
        difference = max(abs(x-y) for x, y in zip(metrics['gain_raw'], old['gain_raw']))
        if difference > .002:
            raise ValueError(f'generalization mismatch: {difference}')
        row = dict(step=step, choices=choices, calibration_scores=scores.tolist(),
                   generalization_replay_max_difference=difference,
                   fixed_block1_steer=old['steer_raw'], **metrics)
        result['history'].append(row)
        torch.save(dict(class_means=means.cpu(), selected_vectors=selected_vectors),
                   a.out_dir / f'step{step:06d}.pt')
        result['complete'] = len(result['history']) == len(rows)
        atomic_json(a.out_dir / 'history.json', result)
        print(json.dumps(dict(step=step, elapsed_seconds=round(time.time()-start, 1),
                              mean_steer=sum(metrics['steer_raw'])/len(choices),
                              mean_gain=sum(metrics['gain_raw'])/len(choices), choices=choices)), flush=True)


if __name__ == '__main__':
    main()
