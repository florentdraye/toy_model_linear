"""Measure residual norms on fixed held-out examples at original checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch

from src.config import ModelConfig
from src.data import enumerate_paths
from src.graph import Graph
from src.model import ToyTransformer
from train_emergence import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run on a GPU compute node')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    source = json.loads((a.run / 'history.json').read_text())
    if source.get('model_transform', {}).get('method', 'identity') != 'identity':
        raise ValueError('This measurement requires original, unaveraged models')
    banks = torch.load(a.run / 'banks.pt', weights_only=True)
    ids = banks['test'][..., 1]
    assert hashlib.sha256(banks['test'].numpy().tobytes()).hexdigest() == source['evaluation_bank_sha256']
    assert not torch.isin(ids, banks['train']).any()
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    assert (paths['nodes'][ids, source['config']['graph_layer']] == torch.tensor(source['latents'])[:, None]).all()
    edges = paths['edge_seqs'][ids.flatten()].cuda()
    model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    model_dir = Path(source.get('model_directory', a.run / 'models'))
    a.out_dir.mkdir(parents=True, exist_ok=False)
    np.save(a.out_dir / 'example_ids.npy', ids.numpy())
    result = dict(complete=False, source=str(a.run), model_transform='identity',
                  definition='mean over fixed held-out examples of Euclidean residual norm; before final LayerNorm',
                  axes=['checkpoint', 'latent', 'depth', 'token'],
                  latents=source['latents'], frequency=source['frequency'],
                  examples_per_latent=ids.shape[1], graph_layer=source['config']['graph_layer'],
                  evaluation_bank_sha256=source['evaluation_bank_sha256'],
                  host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  history=[])
    print(json.dumps({k: result[k] for k in ['host', 'gpu', 'examples_per_latent']}), flush=True)
    atomic_json(a.out_dir / 'history.json', result)
    for row in sorted(source['history'], key=lambda r: r['step']):
        step = row['step']
        model.load_state_dict(torch.load(model_dir / f'step{step:06d}.pt', map_location='cpu', weights_only=True))
        chunks = []
        for batch in edges.split(a.batch_size):
            _, hidden = model(batch, return_hidden=True)
            chunks.append(torch.stack([h.double().norm(dim=-1) for h in hidden], dim=1).cpu())
        norms = torch.cat(chunks).reshape(*ids.shape, len(model.blocks) + 1, model.cfg.seq_len)
        assert torch.isfinite(norms).all()
        result['history'].append(dict(step=step, mean_norm=norms.mean(1).tolist(),
                                      std_norm=norms.std(1).tolist()))
        atomic_json(a.out_dir / 'history.json', result)
        print(f'step {step}: mean block-1 token-2 norm {norms[:, :, 1, 2].mean():.6g}', flush=True)
    result['complete'] = True
    atomic_json(a.out_dir / 'history.json', result)


if __name__ == '__main__':
    main()
