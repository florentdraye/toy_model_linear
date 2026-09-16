"""Independent hook replay, unit-strength LAT, and balanced split-bank stability."""
import argparse
import json
from pathlib import Path
import socket
import time

import torch
from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import forward_at, target_fidelity
from src.graph import Graph
from src.lat import principal_axes
from src.model import ToyTransformer
from train_emergence import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Use allocated GPU')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    start = time.time()
    data = sorted([(json.loads(f.read_text()), f.parent) for f in a.run.glob('step*/history.json')], key=lambda item: item[0]['step'])
    assert len(data) == 21 and all(row['complete'] for row, _ in data)
    assert [row['step'] for row, _ in data] == list(range(0, 20001, 1000))
    source = json.loads((a.source / 'history.json').read_text())
    paths = enumerate_paths(Graph.load(a.source / 'graph.pt'))
    banks = torch.load(a.source / 'banks.pt', weights_only=True)
    test = banks['test']
    off = paths['edge_seqs'][test[..., 0]].cuda()
    labels = paths['labels'][test[..., 1]].cuda()
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    targets = [source['classes'].index(k) for k in source['latents']]
    reference = source['classes'].index(source['reference'])
    result = dict(complete=False, host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  method='Independent forward_at hooks, same fixed held-out pairs; unit strength chosen on existing training calibration scores',
                  rows=[])
    for row, directory in data:
        model.load_state_dict(torch.load(a.source / 'models' / f"step{row['step']:06d}.pt", weights_only=True, map_location='cpu'))
        directions = torch.load(directory / 'directions.pt', weights_only=True, map_location='cpu')
        q = directions['class_codes'].cuda()
        unit_cells = [j for j, cell in enumerate(row['cells']) if cell['alpha'] == 1.]
        scores = torch.tensor(row['calibration_scores'], dtype=torch.float64)
        selected = [unit_cells[j] for j in scores[unit_cells].argmax(0).tolist()]
        choices = [{**row['cells'][j], 'calibration_skill': float(scores[j, k])} for k, j in enumerate(selected)]
        gains, accuracies, replay_gains, replay_accuracies = [], [], [], []
        for k in range(len(targets)):
            for choice, g, acc in [(choices[k], gains, accuracies), (row['choices'][k], replay_gains, replay_accuracies)]:
                depth, pos = choice['depth'], choice['positions']
                delta = choice['alpha'] * (q[depth, targets[k], pos] - q[depth, reference, pos])
                probabilities = forward_at(model, off[k], depth, pos, delta=delta).float().softmax(-1)
                g.append(target_fidelity(probabilities, labels[k])[0])
                acc.append(float((probabilities.argmax(-1) == labels[k]).float().mean()))
        error = max(abs(x - y) for x, y in zip(replay_gains, row['steer_raw']))
        if error > 1e-4: raise ValueError(f'hook replay mismatch at {row["step"]}: {error}')
        if max(abs(x - y) for x, y in zip(replay_accuracies, row['steering_accuracy'])) > 1e-6:
            raise ValueError('hook replay accuracy mismatch')
        result['rows'].append(dict(step=row['step'], choices=choices, unit_steer_raw=gains,
                                   unit_steering_accuracy=accuracies, hook_replay_max_error=error))
        atomic_json(a.run / 'audit.json', result)
        print(json.dumps(dict(step=row['step'], unit_gain=sum(gains)/len(gains), replay_error=error)), flush=True)
    # At the final model, compare axes from two disjoint halves of every
    # negative-class stratum, preserving the balanced negative distribution.
    bank = torch.load(a.run / 'contrast_bank.pt', weights_only=True)
    pairs = bank['pairs']
    ids = pairs.unique(sorted=True)
    lookup = torch.full((len(paths['labels']),), -1, dtype=torch.long); lookup[ids] = torch.arange(len(ids))
    pair_idx = lookup[pairs].cuda()
    pos = source['config']['graph_layer'] - 1
    all_edges = paths['edge_seqs'][ids]
    captured = [torch.empty(len(ids), model.cfg.d_model, device='cuda') for _ in range(2)]
    for first in range(0, len(ids), 1024):
        edges = all_edges[first:first + 1024].cuda()
        h = model.drop(model.tok_emb(edges) + model.pos_emb)
        for d in range(2):
            h = model.blocks[d](h)
            captured[d][first:first + len(edges)] = h[:, pos]
    checks = []
    for d in range(2):
        delta = captured[d][pair_idx[..., 1]] - captured[d][pair_idx[..., 0]]
        stratified = delta.reshape(len(source['classes']), len(source['classes']) - 1, bank['per_negative'], model.cfg.d_model)
        half = bank['per_negative'] // 2
        first, _ = principal_axes(stratified[:, :, :half].flatten(1, 2), seed=9912 + d)
        second, _ = principal_axes(stratified[:, :, half:].flatten(1, 2), seed=9912 + d)
        agreement = (first.double() * second.double()).sum(-1).abs()
        full = directions['axes'][d + 1, :, pos].cuda().double()
        first_full = (first.double() * full).sum(-1).abs()
        checks.append(dict(block=d + 1, pairs_per_half=(len(source['classes']) - 1) * half,
                           half_to_half_abs_cosine=agreement.tolist(),
                           median_half_to_half=float(agreement.median()), minimum_half_to_half=float(agreement.min()),
                           median_half_to_full=float(first_full.median())))
    result.update(complete=True, split_bank_checks=checks, elapsed_seconds=round(time.time()-start, 2))
    atomic_json(a.run / 'audit.json', result)
    print(json.dumps(dict(complete=True, split_checks=[{k: v for k, v in c.items() if k != 'half_to_half_abs_cosine'} for c in checks],
                          elapsed=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__': main()
