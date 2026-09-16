"""Frozen original-checkpoint LAT replay using balanced contrastive class axes."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import torch

from audit_emergence_interventions import score_edit
from scan_emergence_depths import calibration_ids
from src.config import ModelConfig
from src.data import enumerate_paths
from src.graph import Graph
from src.lat import principal_axes
from src.model import ToyTransformer
from train_emergence import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--means-run', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--alphas', type=float, nargs='+', default=[.5, 1., 2.])
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Submit to an allocated GPU compute node')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    start = time.time()
    source = json.loads((a.source / 'history.json').read_text())
    baseline = json.loads((a.means_run / 'history.json').read_text())
    old = next(r for r in baseline['history'] if r['step'] == a.step)
    source_row = next(r for r in source['history'] if r['step'] == a.step)
    bank = torch.load(a.bank, weights_only=True)
    classes = source['classes']
    assert bank['classes'] == classes
    assert bank['source_evaluation_bank_sha256'] == source['evaluation_bank_sha256']
    assert hashlib.sha256(bank['pairs'].numpy().tobytes()).hexdigest() == bank['sha256']
    paths = enumerate_paths(Graph.load(a.source / 'graph.pt'))
    saved = torch.load(a.source / 'banks.pt', weights_only=True)
    train, calibration = calibration_ids(saved, len(paths['labels']))
    allowed = torch.zeros(len(paths['labels']), dtype=torch.bool); allowed[train] = True
    fit = bank['pairs']
    assert allowed[fit].all() and not allowed[saved['test']].any()
    test = saved['test']
    pool = torch.cat([fit.flatten(), calibration.flatten(), test.flatten()]).unique(sorted=True)
    lookup = torch.full((len(paths['labels']),), -1, dtype=torch.long)
    lookup[pool] = torch.arange(len(pool))
    fit_idx, cal_idx, test_idx = [lookup[x].cuda() for x in [fit, calibration, test]]
    edges = paths['edge_seqs'][pool]
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    model.load_state_dict(torch.load(a.source / 'models' / f'step{a.step:06d}.pt', map_location='cpu', weights_only=True))
    hidden = [torch.empty(len(pool), model.cfg.seq_len, model.cfg.d_model, device='cuda') for _ in range(len(model.blocks) + 1)]
    for first in range(0, len(pool), 1024):
        _, states = model(edges[first:first + 1024].cuda(), return_hidden=True)
        for d, state in enumerate(states): hidden[d][first:first + len(state)] = state
    del states
    class_means = torch.load(a.means_run / f'step{a.step:06d}.pt', map_location='cpu', weights_only=True)['class_means'].cuda()
    assert baseline['direction_estimator']['classes'] == classes
    centered_means = (class_means - class_means.mean(1, keepdim=True)).float()
    target = [classes.index(k) for k in source['latents']]
    reference = classes.index(source['reference'])
    labels_cal = paths['labels'][calibration[..., 1]].cuda()
    labels_test = paths['labels'][test[..., 1]].cuda()
    sites = {f'token {j}': [j] for j in range(model.cfg.seq_len)}
    sites['suffix'] = list(range(source['config']['graph_layer'] - 1, model.cfg.seq_len))
    sites['all tokens'] = list(range(model.cfg.seq_len))
    a.out_dir.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, step=a.step, source=str(a.source), means_run=str(a.means_run),
                  model_transform='identity', model_config=source['model_config'], config=source['config'],
                  classes=classes, latents=source['latents'], frequency=source['frequency'], reference=source['reference'],
                  bank_sha256=bank['sha256'], evaluation_bank_sha256=source['evaluation_bank_sha256'],
                  contrast_pairs_per_class=fit.shape[1], per_negative=bank['per_negative'], fit_seed=bank['seed'],
                  estimator='Leading eigenvector of mean(delta delta^T), equivalent to PCA on symmetrized +/- contrasts; per class, depth and token',
                  direction_scale='q_k = u_k * dot(u_k, mu_k-global_class_mean); edit = alpha*(q_target-q_source)',
                  geometry='all 100 per-class unit LAT axes at a common fixed block/token, without reference subtraction',
                  selection='highest training-calibration Brier skill over depth/token layout/alpha; fixed held-out bank untouched',
                  alphas=a.alphas, forward_dtype='FP32, highest matmul precision',
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  original_dom_steer=old['steer_raw'], original_dom_choices=old['choices'],
                  original_gain=old['gain_raw'], eigen_checks=[])
    atomic_json(a.out_dir / 'history.json', result)
    print(json.dumps({key: result[key] for key in ['step', 'host', 'gpu', 'contrast_pairs_per_class']}), flush=True)
    all_axes, all_q, all_pair_mean, eigen_energy = [], [], [], []
    cells, scores = [], []
    for depth in range(len(hidden)):
        differences = hidden[depth][fit_idx[..., 1]] - hidden[depth][fit_idx[..., 0]]
        pair_mean = differences.double().mean(1).float()
        # Treat every token separately, as in the previous per-position mean edit.
        matrix = differences.permute(0, 2, 1, 3).contiguous().reshape(-1, fit.shape[1], model.cfg.d_model)
        axes, info = principal_axes(matrix, seed=9217 + depth)
        axes = axes.reshape(len(classes), model.cfg.seq_len, model.cfg.d_model)
        q = axes * (axes * centered_means[depth]).sum(-1, keepdim=True)
        all_axes.append(axes.cpu()); all_q.append(q.cpu()); all_pair_mean.append(pair_mean.cpu())
        eigen_energy.append(info['explained_energy'].reshape(len(classes), model.cfg.seq_len).cpu())
        result['eigen_checks'].append(dict(depth=depth, max_residual=float(info['residual'].max()),
                                          iterations=info['iterations'], fallback_count=info['fallback_count'],
                                          exact_check_error=info['exact_check_error']))
        del differences, matrix, info
        edit = q[target] - q[reference]
        off = hidden[depth][cal_idx[..., 0]]
        for name, positions in sites.items():
            for alpha in a.alphas:
                score = score_edit(model, off, None, labels_cal, depth, positions, edit, alpha)
                cells.append(dict(depth=depth, site=name, positions=positions, alpha=alpha))
                scores.append(score['skill'])
        print(json.dumps(dict(step=a.step, depth=depth, elapsed=round(time.time()-start, 1),
                              eigen=result['eigen_checks'][-1])), flush=True)
        del off, edit, axes, q
    scores = torch.tensor(scores, dtype=torch.float64)
    assert torch.isfinite(scores).all()
    chosen = scores.argmax(0).tolist()
    choices = [{**cells[c], 'calibration_skill': float(scores[c, j])} for j, c in enumerate(chosen)]
    q = torch.stack(all_q).cuda()
    pair_mean = torch.stack(all_pair_mean).cuda()
    metrics = {key: [] for key in ['steer_raw', 'steering_accuracy', 'patch_raw', 'patch_accuracy',
                                  'random_raw', 'matched_mean_at_lat_site', 'full_dom_at_lat_site']}
    selected_vectors = []
    rng = torch.Generator(device='cuda').manual_seed(991)
    for j, choice in enumerate(choices):
        d, positions, alpha = choice['depth'], choice['positions'], choice['alpha']
        off = hidden[d][test_idx[j:j + 1, :, 0]]
        on = hidden[d][test_idx[j:j + 1, :, 1]]
        label = labels_test[j:j + 1]
        edit = (q[d, target[j]] - q[d, reference]).unsqueeze(0)
        selected_vectors.append((alpha * edit[:, positions]).cpu())
        score = score_edit(model, off, None, label, d, positions, edit, alpha)
        metrics['steer_raw'].extend(score['skill']); metrics['steering_accuracy'].extend(score['accuracy'])
        patch = score_edit(model, off, on, label, d, positions)
        metrics['patch_raw'].extend(patch['skill']); metrics['patch_accuracy'].extend(patch['accuracy'])
        random = torch.randn(edit.shape, device='cuda', generator=rng)
        random *= edit[:, positions].norm() / random[:, positions].norm()
        metrics['random_raw'].extend(score_edit(model, off, None, label, d, positions, random, alpha)['skill'])
        pm = (len(classes) - 1) / len(classes) * (pair_mean[d, target[j]] - pair_mean[d, reference])
        metrics['matched_mean_at_lat_site'].extend(score_edit(model, off, None, label, d, positions, pm[None], 1.)['skill'])
        dm = (class_means[d, target[j]] - class_means[d, reference]).float()[None]
        metrics['full_dom_at_lat_site'].extend(score_edit(model, off, None, label, d, positions, dm, 1.)['skill'])
    test_on = hidden[0][test_idx[..., 1]]
    generalization = score_edit(model, test_on, None, labels_test, 0, [], torch.zeros(len(target), model.cfg.seq_len, model.cfg.d_model, device='cuda'))
    delta_gain = max(abs(x - y) for x, y in zip(generalization['skill'], source_row['gain_raw']))
    if delta_gain > .01: raise ValueError(f'generalization replay mismatch {delta_gain}')
    result.update(complete=True, choices=choices, cells=cells, calibration_scores=scores.tolist(),
                  gain_raw=generalization['skill'], accuracy=generalization['accuracy'],
                  generalization_replay_max_difference=delta_gain,
                  elapsed_seconds=round(time.time() - start, 2), **metrics)
    torch.save(dict(axes=torch.stack(all_axes), class_codes=torch.stack(all_q),
                    paired_mean=torch.stack(all_pair_mean), eigen_energy=torch.stack(eigen_energy),
                    selected_vectors=selected_vectors), a.out_dir / 'directions.pt')
    atomic_json(a.out_dir / 'history.json', result)
    print(json.dumps(dict(complete=True, step=a.step, mean_steer=sum(metrics['steer_raw']) / len(target),
                          mean_steering_accuracy=sum(metrics['steering_accuracy']) / len(target),
                          elapsed=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__': main()
