"""Focused frequency → generalization/steering experiment; GPU work on Condor.

Example: python train_emergence.py --out-dir runs/emergence --steps 6000
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import socket
import subprocess
import time

import torch
import torch.nn.functional as F

from src.config import GraphConfig, ModelConfig
from src.data import enumerate_paths, split_indices, LatentFrequencySampler
from src.graph import Graph
from src.model import ToyTransformer
from src.emergence import paired_bank, fit_directions, measure


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--eval-pairs', type=int, default=1024)
    p.add_argument('--fit-pairs', type=int, default=2048)
    p.add_argument('--eval-batch', type=int, default=2048)
    p.add_argument('--graph-layer', type=int, default=3)
    p.add_argument('--steer-depth', type=int, default=3, help='1-based post-block depth')
    p.add_argument('--site', choices=['token', 'suffix', 'all'], default='suffix')
    p.add_argument('--frequency-ratio', type=float, default=100.)
    p.add_argument('--num-latents', type=int, default=8)
    p.add_argument('--graph-seed', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=314)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-layers', type=int, default=7)
    p.add_argument('--nodes', type=int, default=100)
    p.add_argument('--edges', type=int, default=10)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--n-blocks', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1.)
    p.add_argument('--train-frac', type=float, default=.8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    return p


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    a = parser().parse_args()
    if not (1 <= a.graph_layer < a.n_layers - 1):
        raise ValueError('graph-layer must be intermediate')
    if not 1 <= a.steer_depth <= a.n_blocks:
        raise ValueError('steer-depth must name an existing block')
    if not 0 < a.train_frac < 1:
        raise ValueError('train-frac must be strictly between zero and one')
    if min(a.steps, a.eval_every, a.batch_size, a.eval_batch, a.num_latents) < 1 or min(a.eval_pairs, a.fit_pairs) < 2:
        raise ValueError('counts must be positive; pair counts must be >= 2')
    if a.edges ** (a.n_layers - 1) > 5_000_000:
        raise ValueError('enumerated support exceeds 5M paths; reduce graph size')
    if a.device == 'cpu' and a.steps > 2:
        raise ValueError('CPU is for smoke checks only (at most two steps); submit GPU work')
    if (a.out_dir / 'history.json').exists() and not a.resume:
        raise FileExistsError('output exists; use a fresh directory or --resume')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision('high')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    gcfg = GraphConfig(a.n_layers, (1,) + (a.nodes,) * (a.n_layers - 1), a.edges, a.graph_seed)
    graph = Graph(gcfg)
    paths = enumerate_paths(graph)
    train_idx, test_idx = split_indices(len(paths['labels']), a.train_frac, a.data_seed)
    allowed = torch.zeros(len(paths['labels']), dtype=torch.bool)
    allowed[train_idx] = True
    train_edges = paths['edge_seqs'][train_idx].to(a.device)
    train_nodes = paths['nodes'][train_idx].to(a.device)
    sampler = LatentFrequencySampler(train_nodes[:, a.graph_layer], a.frequency_ratio, a.data_seed)
    # Selection depends on frequency rank only, never on measured curves.
    order = sampler.order.cpu()
    ranks = torch.linspace(1, len(order) - 1, a.num_latents).round().long().unique()
    reference = int(sampler.classes[order[0]])
    targets = sampler.classes[order[ranks].to(a.device)].tolist()
    fit = paired_bank(paths, allowed, a.graph_layer, targets, reference,
                      a.fit_pairs, a.data_seed + 1, a.edges)
    test = paired_bank(paths, ~allowed, a.graph_layer, targets, reference,
                       a.eval_pairs, a.data_seed + 2, a.edges)
    torch.save({'fit': fit['ids'], 'test': test['ids'], 'train': train_idx,
                'test_support': test_idx}, a.out_dir / 'banks.pt')
    fit = {k: v.to(a.device) for k, v in fit.items()}
    test = {k: v.to(a.device) for k, v in test.items()}
    positions = ([a.graph_layer - 1] if a.site == 'token' else
                 list(range(a.graph_layer - 1 if a.site == 'suffix' else 0, a.n_layers - 1)))
    mcfg = ModelConfig(vocab_size=a.edges, seq_len=a.n_layers - 1, n_classes=a.nodes,
                       d_model=a.d_model, n_heads=4, d_ff=4 * a.d_model,
                       n_blocks=a.n_blocks, mlp_activation='relu2')
    model = ToyTransformer(mcfg).to(a.device)
    groups = [{'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': a.weight_decay},
              {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(.9, .95), fused=a.device.startswith('cuda'))
    gen = torch.Generator(device=a.device).manual_seed(a.seed + 1234)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    result = {'config': config, 'model_config': asdict(mcfg), 'graph_config': asdict(gcfg),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'host': socket.gethostname(), 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name() if a.device.startswith('cuda') else 'cpu',
              'reference': reference, 'latents': targets, 'positions': positions,
              'frequency': sampler.probabilities[order[ranks].to(a.device)].tolist(),
              'classes': sampler.classes.tolist(), 'probabilities': sampler.probabilities.tolist(),
              'history': []}
    start = 0
    counts = torch.zeros(len(sampler.classes), device=a.device, dtype=torch.long)
    if a.resume:
        saved = torch.load(a.out_dir / 'resume.pt', map_location=a.device, weights_only=False)
        for k, v in config.items():
            if k not in ('resume', 'device', 'out_dir') and saved['result']['config'][k] != v:
                raise ValueError(f'cannot resume with changed {k}')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        gen.set_state(saved['sampler_rng'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        counts = saved['counts']
        result = saved['result']
        start = saved['step']
    graph.save(a.out_dir / 'graph.pt')
    print(json.dumps({k: result[k] for k in ('host', 'gpu', 'latents', 'frequency', 'reference', 'positions')}), flush=True)
    t0 = time.time()
    running_loss = torch.zeros((), device=a.device)
    loss_n = 0

    def checkpoint(step):
        nonlocal running_loss, loss_n
        model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=a.device.startswith('cuda')):
            vectors = fit_directions(model, fit, a.steer_depth, positions, a.eval_batch)
            metrics = measure(model, test, vectors, a.steer_depth, positions, a.eval_batch)
        rec = {'step': step, 'examples_seen': step * a.batch_size,
               'latent_counts': counts[order[ranks].to(a.device)].tolist(),
               'train_loss': float(running_loss / max(loss_n, 1)), **metrics}
        result['history'].append(rec)
        result['complete'] = step == a.steps
        atomic_json(a.out_dir / 'history.json', result)
        (a.out_dir / 'directions').mkdir(exist_ok=True)
        torch.save(vectors.cpu(), a.out_dir / 'directions' / f'step{step:06d}.pt')
        state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step,
                 'sampler_rng': gen.get_state(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],
                 'counts': counts, 'result': result}
        torch.save(state, a.out_dir / 'resume.tmp')
        (a.out_dir / 'resume.tmp').replace(a.out_dir / 'resume.pt')
        print(f"step {step:5d} loss {rec['train_loss']:.4f} "
              f"gain {sum(metrics['gain_raw'])/len(targets):.3f} "
              f"steer {sum(metrics['steer_raw'])/len(targets):.3f} "
              f"patch {sum(metrics['patch_raw'])/len(targets):.3f} "
              f"acc {sum(metrics['accuracy'])/len(targets):.3f} ({time.time()-t0:.0f}s)", flush=True)
        running_loss.zero_()
        loss_n = 0
        model.train()

    if not a.resume:
        checkpoint(0)
    for step in range(start + 1, a.steps + 1):
        idx, cls = sampler.sample(a.batch_size, gen)
        counts += torch.bincount(cls, minlength=len(counts))
        # Constant learning rate after short fixed warmup: no schedule-induced
        # simultaneous transition, and extending a planned horizon changes no LR.
        lr = a.lr * min(step / 100, 1.)
        for group in opt.param_groups:
            group['lr'] = lr
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=a.device.startswith('cuda')):
            logits = model(train_edges[idx])
            loss = F.cross_entropy(logits, train_nodes[idx, -1])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite training loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        running_loss += loss.detach()
        loss_n += 1
        if step % a.eval_every == 0 or step == a.steps:
            checkpoint(step)
    from plot_emergence import plot
    plot([a.out_dir / 'history.json'], a.out_dir / 'figures')


if __name__ == '__main__':
    main()
