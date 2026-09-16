"""Build the fixed train-only LAT contrast bank (CPU data preparation only)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from src.graph import Graph
from src.data import enumerate_paths
from src.lat import balanced_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-history', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--banks', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-negative', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(2)
    source = json.loads(a.source_history.read_text())
    paths = enumerate_paths(Graph.load(a.graph))
    banks = torch.load(a.banks, weights_only=True)
    allowed = torch.zeros(len(paths['labels']), dtype=torch.bool)
    allowed[banks['train']] = True
    pairs = balanced_pairs(paths, allowed, source['config']['graph_layer'], source['classes'], a.per_negative)
    assert allowed[pairs].all() and not allowed[banks['test']].any()
    assert torch.equal(paths['edge_seqs'][pairs[..., 0]][..., 3:], paths['edge_seqs'][pairs[..., 1]][..., 3:])
    labels = paths['nodes'][pairs, source['config']['graph_layer']]
    for j, k in enumerate(source['classes']):
        assert (labels[j, :, 1] == k).all()
        values, counts = labels[j, :, 0].unique(return_counts=True)
        assert len(values) == len(source['classes']) - 1 and (counts == a.per_negative).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(pairs=pairs, classes=source['classes'], per_negative=a.per_negative,
                    seed=9211, sha256=hashlib.sha256(pairs.numpy().tobytes()).hexdigest(),
                    source_evaluation_bank_sha256=source['evaluation_bank_sha256']), a.out)
    print(json.dumps(dict(shape=list(pairs.shape), unique_paths=len(pairs.unique()),
                          unique_positive_paths_range=[min(len(p[:, 1].unique()) for p in pairs), max(len(p[:, 1].unique()) for p in pairs)],
                          sha256=hashlib.sha256(pairs.numpy().tobytes()).hexdigest())))


if __name__ == '__main__': main()
