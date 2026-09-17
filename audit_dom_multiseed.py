"""Independent hook replay of saved selected DoM vectors on real GPU checkpoints."""
import argparse
import json
from pathlib import Path
import socket
import torch

from src.config import ModelConfig
from src.data import enumerate_paths
from src.emergence import forward_at, target_fidelity
from src.graph import Graph
from src.model import ToyTransformer


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--steps', type=int, nargs='+', default=[0, 8000, 20000])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('run on an allocated GPU node')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    checks = []
    for directory in a.runs:
        source = json.loads((directory/'history.json').read_text())
        paths = enumerate_paths(Graph.load(directory/'graph.pt'))
        bank = torch.load(directory/'banks.pt', weights_only=True)
        ids = bank['test']
        allowed = torch.zeros(len(paths['labels']), dtype=torch.bool)
        allowed[bank['train']] = True
        assert not allowed[ids].any() and allowed[bank['fit']].all()
        model = ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
        rows = {r['step']: r for r in source['history']}
        for step in a.steps:
            model.load_state_dict(torch.load(directory/'models'/f'step{step:06d}.pt',
                                             map_location='cpu', weights_only=True))
            vectors = torch.load(directory/'directions'/f'step{step:06d}.pt',
                                 map_location='cuda', weights_only=True)
            row = rows[step]
            max_gain_error, max_accuracy_error = 0., 0.
            for j, cell in enumerate(row['choices']):
                off = paths['edge_seqs'][ids[j, :, 0]].cuda()
                labels = paths['labels'][ids[j, :, 1]].cuda()
                logits = forward_at(model, off, cell['depth'], cell['positions'],
                                    vectors[j, cell['positions']])
                gain, _ = target_fidelity(logits.softmax(-1), labels)
                accuracy = float((logits.argmax(-1) == labels).float().mean())
                max_gain_error = max(max_gain_error, abs(gain-row['steer_raw'][j]))
                max_accuracy_error = max(max_accuracy_error, abs(accuracy-row['steering_accuracy'][j]))
            check = dict(seed=source['config']['seed'], step=step,
                         max_gain_error=max_gain_error, max_accuracy_error=max_accuracy_error)
            # Batch grouping can change FP32 roundoff. This is not a BF16 comparison.
            if max_gain_error > 1e-4 or max_accuracy_error > 1/ids.shape[1]+1e-6:
                raise AssertionError(check)
            checks.append(check)
            print(json.dumps(check), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                                    checks=checks, passed=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
