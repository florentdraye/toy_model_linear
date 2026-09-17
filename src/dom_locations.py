"""Full-support DoM, training-only location selection, fixed held-out scoring."""
import torch

from .emergence import from_hidden, target_fidelity
from .uniform_means import hidden_at


@torch.no_grad()
def all_means(model, bank, batch_size):
    device = next(model.parameters()).device
    means = torch.zeros(len(model.blocks)+1, len(bank.classes), model.cfg.seq_len,
                        model.cfg.d_model, device=device, dtype=torch.float64)
    start = 0
    for j, count in enumerate(bank.counts):
        for edges in bank.edges[start:start+count].split(batch_size):
            h = hidden_at(model, edges, 0)
            means[0, j] += h.double().sum(0)
            for depth, block in enumerate(model.blocks, 1):
                h = block(h)
                means[depth, j] += h.double().sum(0)
        means[:, j] /= count
        start += count
    return means


def capture(model, edges, depth, batch_size):
    h = torch.cat([hidden_at(model, e, depth)
                   for e in edges.flatten(0, 1).split(batch_size)])
    return h.reshape(*edges.shape[:2], *h.shape[1:])


def score(model, hidden, labels, depth, batch_size):
    logits = torch.cat([from_hidden(model, h, depth)
                        for h in hidden.flatten(0, 1).split(batch_size)])
    p = logits.float().softmax(-1).reshape(*labels.shape, -1)
    values = [target_fidelity(x, y) for x, y in zip(p, labels)]
    return dict(raw=[x[0] for x in values], se=[x[1] for x in values],
                accuracy=(p.argmax(-1) == labels).float().mean(-1).tolist())


@torch.no_grad()
def measure_locations(model, bank, targets, reference, calibration, test,
                      graph_layer, mean_batch=8192, eval_batch=2048):
    """Same 56 cells as the archived scan; strength 1; all forwards FP32.

    The bank may omit unused classes: each target/reference mean still uses
    its ENTIRE training support. No random draws or changes to model state.
    Selection is completed before reading any test responses.
    """
    if model.training:
        raise ValueError('measurement requires model.eval()')
    device = next(model.parameters()).device
    with torch.autocast(device.type, enabled=False):
        means = all_means(model, bank, mean_batch)
        ix = [bank.classes.index(k) for k in targets]
        vectors = (means[:, ix] - means[:, bank.classes.index(reference):
                                       bank.classes.index(reference)+1]).float()
        sites = {f'token {p}': [p] for p in range(model.cfg.seq_len)}
        sites['suffix'] = list(range(graph_layer-1, model.cfg.seq_len))
        sites['all tokens'] = list(range(model.cfg.seq_len))
        cells, scores = [], []
        for depth in range(len(model.blocks)+1):
            h = capture(model, calibration['off'], depth, eval_batch)
            for name, positions in sites.items():
                edited = h.clone()
                edited[:, :, positions] += vectors[depth, :, None, positions]
                scores.append(score(model, edited, calibration['y_on'], depth, eval_batch)['raw'])
                cells.append(dict(depth=depth, site=name, positions=positions, alpha=1.))
        scores = torch.tensor(scores, dtype=torch.float64)
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite calibration scores')
        chosen = scores.argmax(0).tolist()
        choices = [{**cells[c], 'calibration_skill': float(scores[c, j])}
                   for j, c in enumerate(chosen)]
        result = dict(choices=choices, calibration_scores=scores.tolist())
        # All test methods use the same FP32 continuation and examples.
        clean = capture(model, test['on'], len(model.blocks), eval_batch)
        m = score(model, clean, test['y_on'], len(model.blocks), eval_batch)
        result.update(gain_raw=m['raw'], gain_se=m['se'], accuracy=m['accuracy'])
        n = len(targets)
        for key in ('steer_raw', 'steer_se', 'steering_accuracy', 'patch_raw', 'patch_accuracy'):
            result[key] = [None] * n
        selected = torch.zeros(n, model.cfg.seq_len, model.cfg.d_model, device=device)
        for c in sorted(set(chosen)):
            js = [j for j, value in enumerate(chosen) if value == c]
            depth, pos = cells[c]['depth'], cells[c]['positions']
            off = capture(model, test['off'][js], depth, eval_batch)
            on = capture(model, test['on'][js], depth, eval_batch)
            edited = off.clone()
            edited[:, :, pos] += vectors[depth, js][:, None, pos]
            steered = score(model, edited, test['y_on'][js], depth, eval_batch)
            off[:, :, pos] = on[:, :, pos]
            patched = score(model, off, test['y_on'][js], depth, eval_batch)
            for k, j in enumerate(js):
                selected[j, pos] = vectors[depth, j, pos]
                for key, value in [('steer_raw', steered['raw']), ('steer_se', steered['se']),
                                   ('steering_accuracy', steered['accuracy']),
                                   ('patch_raw', patched['raw']), ('patch_accuracy', patched['accuracy'])]:
                    result[key][j] = value[k]
        result['vector_norm'] = selected.flatten(1).norm(dim=1).tolist()
    return result, selected, means
