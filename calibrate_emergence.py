"""Choose a fixed intervention site using calibration validation only.

Run on a GPU allocation, after a pilot. The sweep does not load the test bank.
Its chosen site must be fixed before a fresh set of training seeds is measured.
"""
import argparse
import json
from pathlib import Path
import torch
from src.config import ModelConfig
from src.model import ToyTransformer
from src.graph import Graph
from src.data import enumerate_paths
from src.emergence import fit_directions, optimize_directions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--sites', nargs='+', default=['token', 'suffix', 'all'])
    p.add_argument('--radii', type=float, nargs='+', default=[1., 3.])
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    state = torch.load(a.run / 'resume.pt', map_location='cpu', weights_only=False)
    cfg = state['result']['config']
    model = ToyTransformer(ModelConfig(**state['result']['model_config'])).cuda().eval()
    model.load_state_dict(state['model'])
    paths = enumerate_paths(Graph.load(a.run / 'graph.pt'))
    ids = torch.load(a.run / 'banks.pt', weights_only=True)['fit']
    bank = {'off': paths['edge_seqs'][ids[..., 0]].cuda(),
            'on': paths['edge_seqs'][ids[..., 1]].cuda(),
            'y_on': paths['labels'][ids[..., 1]].cuda()}
    records = []
    for depth in a.depths:
        for site in a.sites:
            positions = ([cfg['graph_layer']-1] if site == 'token' else
                         list(range(cfg['graph_layer']-1 if site == 'suffix' else 0, cfg['n_layers']-1)))
            for scale in a.radii:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    means = fit_directions(model, bank, depth, positions)
                    vectors, diagnostics = optimize_directions(model, bank, means, depth, positions,
                                                                 steps=a.steps, radius_scale=scale)
                scores = [1-v/(1-1/cfg['nodes']) for v in diagnostics['direction_validation_after']]
                record = {'depth': depth, 'site': site, 'radius_scale': scale,
                          'validation_skill': scores, 'mean_validation_skill': sum(scores)/len(scores),
                          **diagnostics}
                records.append(record)
                print(json.dumps(record), flush=True)
                (a.run / 'site_calibration.json').write_text(json.dumps(records, indent=2) + '\n')
    best = max(records, key=lambda r: r['mean_validation_skill'])
    print('BEST BY CALIBRATION VALIDATION: ' + json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
