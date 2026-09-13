"""Separate mean definition, token location, and strength on frozen checkpoints.

Calibration sweep uses training pairs only. Fixed held-out comparisons at block
1 are reported separately; no best-test selection. Mean vectors remain analytic.
An independent FP32 continuation bypasses intervention hooks and autocast.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess

import torch

from scan_emergence_depths import calibration_ids
from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import from_hidden, forward_at, target_fidelity
from src.graph import Graph
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank, hidden_at
from train_emergence import atomic_json


@torch.no_grad()
def capture(model, paths, ids, depth, batch=2048):
    shape = ids.shape
    edges = paths['edge_seqs'][ids.flatten()]
    h = torch.cat([hidden_at(model, e.cuda(), depth) for e in edges.split(batch)])
    return h.reshape(*shape, *h.shape[1:])


@torch.no_grad()
def score_edit(model, off, on, labels, depth, positions, vector=None, alpha=1.):
    edited = off.clone()
    if vector is None:
        edited[:, :, positions] = on[:, :, positions]
    else:
        edited[:, :, positions] += alpha * vector[:, None, positions]
    logits = torch.cat([from_hidden(model, h, depth) for h in edited.flatten(0, 1).split(2048)])
    probabilities = logits.softmax(-1).reshape(*labels.shape, -1)
    skill = [target_fidelity(p, y)[0] for p, y in zip(probabilities, labels)]
    accuracy = (probabilities.argmax(-1) == labels).float().mean(-1).tolist()
    return dict(skill=skill, accuracy=accuracy, mean_skill=sum(skill)/len(skill),
                mean_accuracy=sum(accuracy)/len(accuracy))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('submit this audit to a GPU compute node')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    source = json.loads((a.run / 'history.json').read_text())
    if not source['complete']:
        raise ValueError('source run must be complete')
    step = source['history'][-1]['step']
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    saved = torch.load(a.run / 'banks.pt', weights_only=True)
    train, validation = calibration_ids(saved, len(paths['labels']))
    fitting = saved['fit'][:, :max(1, int(saved['fit'].shape[1]*.75))]
    allowed = torch.zeros(len(paths['labels']), dtype=torch.bool)
    allowed[train] = True
    if not allowed[fitting].all():
        raise ValueError('fitting pairs must be in training support')
    cfg = source['config']
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    model.load_state_dict(torch.load(a.run / 'models' / f'step{step:06d}.pt',
                                    map_location='cpu', weights_only=True))
    bank = UniformMeanBank(paths['edge_seqs'][train].cuda(),
                          paths['nodes'][train, cfg['graph_layer']].cuda())
    positions = list(range(model.cfg.seq_len))
    target_ix = [bank.classes.index(k) for k in source['latents']]
    ref_ix = bank.classes.index(source['reference'])
    sites = {f'token{j}': [j] for j in positions}
    sites['suffix'] = positions[cfg['graph_layer']-1:]
    sites['all'] = positions
    a.out_dir.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, source=str(a.run), step=step, seed=cfg['seed'],
                  latents=source['latents'], reference=source['reference'],
                  graph_layer=cfg['graph_layer'], sites=sites, host=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(), forward_dtype='float32',
                  mean_support=bank.description(source['latents']),
                  calibration='last quarter of training pairs; overlaps full-support means',
                  matched_mean_support='first three quarters of training pairs',
                  fixed_test_protocol='block 1; token2, last, suffix; three means; alpha 1',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  checks=[], records=[], test_records=[])
    torch.save(dict(calibration=validation, fitting=fitting), a.out_dir / 'pair_ids.pt')
    print(json.dumps({k: result[k] for k in ('host', 'gpu', 'seed')}), flush=True)
    for depth in range(1, len(model.blocks)+1):
        dom, means = bank.fit(model, source['latents'], depth, positions)
        off = capture(model, paths, validation[..., 0], depth)
        on = capture(model, paths, validation[..., 1], depth)
        fit_off = capture(model, paths, fitting[..., 0], depth)
        fit_on = capture(model, paths, fitting[..., 1], depth)
        vectors = dict(dom=dom, reference=(means[target_ix]-means[ref_ix]).float(),
                       matched=(fit_on.double()-fit_off.double()).mean(1).float())
        del fit_off, fit_on
        labels = paths['labels'][validation[..., 1]].cuda()
        # Independent implementation: compare continuation and hook-based edits.
        edges = paths['edge_seqs'][validation[0, :16, 0]].cuda()
        h = hidden_at(model, edges, depth)
        direct = model(edges)
        resumed = from_hidden(model, h, depth)
        changed = h.clone()
        changed[:, -1] += dom[0, -1]
        hook_logits = forward_at(model, edges, depth, [positions[-1]], delta=dom[0, -1:])
        torch.testing.assert_close(resumed, direct, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(from_hidden(model, changed, depth), hook_logits, atol=2e-5, rtol=1e-5)
        result['checks'].append(dict(depth=depth,
            continuation_max_error=float((resumed-direct).abs().max()),
            hook_max_error=float((from_hidden(model, changed, depth)-hook_logits).abs().max())))
        for site, pos in sites.items():
            result['records'].append(dict(depth=depth, site=site, method='patch', alpha=1.,
                **score_edit(model, off, on, labels, depth, pos)))
            for name, vector in vectors.items():
                for alpha in (1., 2., 4., 8., 16.):
                    result['records'].append(dict(depth=depth, site=site, method=name, alpha=alpha,
                        **score_edit(model, off, on, labels, depth, pos, vector, alpha)))
        clean = score_edit(model, off, on, labels, depth, positions)
        baseline = score_edit(model, off, on, labels, depth, [], dom)
        result['records'].extend([dict(depth=depth, site='all', method='clean', alpha=0., **clean),
                                  dict(depth=depth, site='all', method='baseline', alpha=0., **baseline)])
        # Fixed comparisons motivated by the archived successful mean control.
        if depth == 1:
            test_off = capture(model, paths, saved['test'][..., 0], depth)
            test_on = capture(model, paths, saved['test'][..., 1], depth)
            test_labels = paths['labels'][saved['test'][..., 1]].cuda()
            for site in (f"token{cfg['graph_layer']-1}", f'token{positions[-1]}', 'suffix'):
                pos = sites[site]
                for name, vector in {'patch': None, **vectors}.items():
                    result['test_records'].append(dict(depth=depth, site=site, method=name, alpha=1.,
                        **score_edit(model, test_off, test_on, test_labels, depth, pos, vector)))
            del test_off, test_on
        torch.save(dict(vectors={k: v.cpu() for k, v in vectors.items()}, class_means=means.cpu()),
                   a.out_dir / f'depth{depth}.pt')
        result['complete'] = depth == len(model.blocks)
        atomic_json(a.out_dir / 'audit.json', result)
        rows = [r for r in result['records'] if r['depth']==depth and r['method']!='patch' and r['alpha']>0]
        best = max(rows, key=lambda r: r['mean_skill'])
        print(json.dumps(dict(depth=depth, best_calibration=best)), flush=True)


if __name__ == '__main__':
    main()
