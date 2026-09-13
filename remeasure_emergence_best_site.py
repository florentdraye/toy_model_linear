"""Scan all single-token sites and report calibration-selected held-out steering.

Each model/latent/checkpoint selects its depth and position independently.
Two analytic contrasts (target-rest and target-reference) are reported at unit
strength and with a separate finite strength grid. No direction optimization.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

import torch
import torch.nn.functional as F

from audit_emergence_interventions import capture
from scan_emergence_depths import calibration_ids
from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import from_hidden, measure
from src.graph import Graph
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank
from train_emergence import atomic_json


@torch.no_grad()
def all_depth_means(model, bank, batch_size=8192):
    """Full-support FP64 class means at every block/token in one FP32 pass.

    Returns [depth, class, token, width]; avoids recomputing shared prefixes.
    """
    if model.training or batch_size < 1:
        raise ValueError('need an eval model and positive batch size')
    device = next(model.parameters()).device
    means = torch.zeros(len(model.blocks)+1, len(bank.classes), model.cfg.seq_len,
                        model.cfg.d_model, dtype=torch.float64, device=device)
    start = 0
    with torch.autocast(device.type, enabled=False):
        for j, count in enumerate(bank.counts):
            for edges in bank.edges[start:start+count].split(batch_size):
                h = model.drop(model.tok_emb(edges.to(device)) + model.pos_emb)
                means[0, j] += h.double().sum(0)
                for depth, block in enumerate(model.blocks, start=1):
                    h = block(h)
                    means[depth, j] += h.double().sum(0)
            means[:, j] /= count
            start += count
    return means


def choose_cells(cube, alphas, unit_only=False):
    """Choose from [depth, token, strength, latent] CALIBRATION scores only.

    Deterministic tie order: earliest depth, earliest token, smallest strength.
    No pooling across seeds/latents before selection, and no score clipping.
    """
    if cube.ndim != 4 or cube.shape[2] != len(alphas) or not torch.isfinite(cube).all():
        raise ValueError('need a finite calibration cube matching the strength grid')
    use = [alphas.index(1.)] if unit_only else list(range(len(alphas)))
    scores = cube[:, :, use]
    flat = scores.flatten(0, 2)
    best = flat.argmax(0)
    choices = []
    for latent, i in enumerate(best.tolist()):
        depth, rem = divmod(i, scores.shape[1]*len(use))
        position, ai = divmod(rem, len(use))
        choices.append(dict(depth=depth, position=position, alpha=alphas[use[ai]],
                            calibration_skill=float(flat[i, latent])))
    return choices


@torch.no_grad()
def calibration_scores(model, hidden, labels, vector, depth, position, alpha):
    h = hidden.clone()
    h[:, :, position] += alpha * vector[:, None, position]
    logits = torch.cat([from_hidden(model, part, depth) for part in h.flatten(0, 1).split(2048)])
    p = logits.float().softmax(-1).reshape(*labels.shape, -1).double()
    target = F.one_hot(labels, p.shape[-1])
    return (1 - (p-target).square().sum(-1).mean(-1)/(1-1/p.shape[-1])).cpu()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--alphas', type=float, nargs='+', default=[1., 2., 4., 8., 16.])
    p.add_argument('--steps', type=int, nargs='+', help='optional subset; default all saved checkpoints')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('submit this replay to a GPU compute node')
    if 1. not in a.alphas or any(x <= 0 for x in a.alphas) or len(set(a.alphas)) != len(a.alphas):
        raise ValueError('need distinct positive strengths including 1')
    a.alphas.sort()
    source = json.loads((a.run / 'history.json').read_text())
    if not source.get('complete'):
        raise ValueError('source training must be complete')
    rows = [r for r in source['history'] if a.steps is None or r['step'] in a.steps]
    if not rows or (a.steps is not None and set(a.steps) != {r['step'] for r in rows}):
        raise ValueError('requested checkpoints unavailable')
    for row in rows:
        if not (a.run / 'models' / f"step{row['step']:06d}.pt").is_file():
            raise FileNotFoundError(f"missing model at {row['step']}")
    a.out_dir.mkdir(parents=True, exist_ok=False)
    for name in ('calibration', 'directions', 'class_means'):
        (a.out_dir / name).mkdir()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    saved = torch.load(a.run / 'banks.pt', weights_only=True)
    train, pairs = calibration_ids(saved, len(paths['labels']))
    bank = UniformMeanBank(paths['edge_seqs'][train].cuda(),
                          paths['nodes'][train, source['config']['graph_layer']].cuda())
    ix = [bank.classes.index(k) for k in source['latents']]
    ref_ix = bank.classes.index(source['reference'])
    test_ids = saved['test']
    test = dict(off=paths['edge_seqs'][test_ids[..., 0]].cuda(),
                on=paths['edge_seqs'][test_ids[..., 1]].cuda(),
                y_off=paths['labels'][test_ids[..., 0]].cuda(),
                y_on=paths['labels'][test_ids[..., 1]].cuda())
    labels = paths['labels'][pairs[..., 1]].cuda()
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    result = dict(complete=False, source=str(a.run), seed=source['config']['seed'],
                  latents=source['latents'], frequency=source['frequency'], reference=source['reference'],
                  graph_layer=source['config']['graph_layer'], model_config=source['model_config'],
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(), alphas=a.alphas,
                  selection='per checkpoint, model seed, and latent; calibration argmax over single-token sites',
                  calibration='last quarter of original training calibration pairs; overlaps mean support',
                  evaluation='original held-out pairs; never used for selection',
                  mean_estimator=bank.description(source['latents']),
                  reference_mean_estimator=bank.description(source['latents'], source['reference']),
                  calibration_dtype='float32', evaluation_dtype='bfloat16 autocast',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  history=[])
    torch.save(dict(calibration=pairs, test=test_ids, train=train), a.out_dir / 'pair_ids.pt')
    print(json.dumps({k: result[k] for k in ('host', 'gpu', 'seed', 'selection')}), flush=True)
    start = time.time()
    for old in rows:
        step = old['step']
        model.load_state_dict(torch.load(a.run / 'models' / f'step{step:06d}.pt',
                                        map_location='cpu', weights_only=True))
        means = all_depth_means(model, bank)
        positive = means[:, ix]
        vectors = dict(rest=(positive-(means.sum(1, keepdim=True)-positive)/(len(bank.classes)-1)).float(),
                       reference=(positive-means[:, ref_ix:ref_ix+1]).float())
        cubes = {key: torch.empty(len(model.blocks)+1, model.cfg.seq_len, len(a.alphas), len(ix),
                                  dtype=torch.float64) for key in vectors}
        for depth in range(len(model.blocks)+1):
            hidden = capture(model, paths, pairs[..., 0], depth)
            for name, v in vectors.items():
                for pos in range(model.cfg.seq_len):
                    for ai, alpha in enumerate(a.alphas):
                        cubes[name][depth, pos, ai] = calibration_scores(
                            model, hidden, labels, v[depth], depth, pos, alpha)
        selections = {}
        # Freeze ALL four selections before scoring any held-out pairs.
        for name, cube in cubes.items():
            for mode in ('unit', 'scaled'):
                selections[f'{name}_{mode}'] = choose_cells(cube, a.alphas, mode=='unit')
        row = dict(step=step, gain_raw=old['gain_raw'], gain_se=old['gain_se'], groups={})
        selected_vectors = {}
        max_difference = 0.
        for name, choices in selections.items():
            direction = vectors[name.split('_')[0]]
            metrics = {}
            selected = []
            for j, choice in enumerate(choices):
                depth, pos, alpha = choice['depth'], choice['position'], choice['alpha']
                vector = alpha * direction[depth, j:j+1, pos:pos+1]
                selected.append(vector[0].cpu())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    measured = measure(model, {k: v[j:j+1] for k, v in test.items()},
                                       vector, depth, [pos], source['config']['eval_batch'])
                max_difference = max(max_difference, abs(measured['gain_raw'][0]-old['gain_raw'][j]))
                for key, value in measured.items():
                    metrics.setdefault(key, []).extend(value)
            row['groups'][name] = dict(choices=choices, **metrics)
            selected_vectors[name] = torch.stack(selected)
        if max_difference > .002:
            raise ValueError(f'generalization replay mismatch at {step}: {max_difference}')
        row['generalization_replay_max_difference'] = max_difference
        result['history'].append(row)
        name = f'step{step:06d}.pt'
        torch.save(cubes, a.out_dir / 'calibration' / name)
        torch.save(dict(vectors=selected_vectors, selections=selections), a.out_dir / 'directions' / name)
        torch.save(means.cpu(), a.out_dir / 'class_means' / name)
        result['complete'] = len(result['history']) == len(rows)
        atomic_json(a.out_dir / 'history.json', result)
        print(json.dumps(dict(step=step, elapsed_seconds=round(time.time()-start, 1),
                              **{k: sum(v['steer_raw'])/len(ix) for k, v in row['groups'].items()})), flush=True)


if __name__ == '__main__':
    main()
