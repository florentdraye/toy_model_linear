"""Matched causal measurements at one fixed transformer depth.

Each latent k is compared with the same frequent reference r. Prefixes reach
k or r; their suffix edge choices are identical. The graph, not the network,
defines the output difference. Directions are fitted on training paths only.
"""
import torch
import torch.nn.functional as F


def path_ids(edges, radix):
    powers = radix ** torch.arange(edges.shape[-1] - 1, -1, -1,
                                   device=edges.device)
    return (edges * powers).sum(-1)


def paired_bank(paths, allowed, layer, targets, reference, n, seed, radix):
    """Fixed unique pairs; BOTH paths belong to the requested support.

    Rejection is uniform over prefix pairs and suffixes. Do not discard pairs
    whose endpoints coincide: their zero teacher effect penalizes leakage.
    """
    rng = torch.Generator().manual_seed(seed)
    edges, nodes = paths['edge_seqs'], paths['nodes']
    prefixes = edges[:, :layer].unique(dim=0)
    # Enumeration is lexicographic; a zero suffix identifies each prefix node.
    suffix_count = radix ** (edges.shape[1] - layer)
    prefix_nodes = nodes[::suffix_count, layer]
    by_node = {int(k): prefixes[prefix_nodes == k] for k in [reference, *targets]}
    pairs = []
    for k in targets:
        collected, keys = [], set()
        for _ in range(1000):
            size = max(4096, 4 * (n - len(keys)))
            a = by_node[reference][torch.randint(len(by_node[reference]), (size,), generator=rng)]
            b = by_node[k][torch.randint(len(by_node[k]), (size,), generator=rng)]
            suffix = torch.randint(radix, (size, edges.shape[1] - layer), generator=rng)
            off, on = torch.cat([a, suffix], 1), torch.cat([b, suffix], 1)
            ia, ib = path_ids(off, radix), path_ids(on, radix)
            good = allowed[ia] & allowed[ib]
            for i, j in zip(ia[good].tolist(), ib[good].tolist()):
                if (i, j) not in keys:
                    keys.add((i, j))
                    collected.append((i, j))
                    if len(collected) == n:
                        break
            if len(collected) == n:
                break
        if len(collected) != n:
            raise ValueError(f"insufficient unique matched pairs for latent {k}: {len(collected)}/{n}")
        pairs.append(torch.tensor(collected))
    ids = torch.stack(pairs)
    return {"off": edges[ids[..., 0]], "on": edges[ids[..., 1]],
            "y_off": nodes[ids[..., 0], -1], "y_on": nodes[ids[..., 1], -1],
            "ids": ids}


def forward_at(model, edges, depth, positions, delta=None, capture=False):
    """Post-block intervention with a temporary hook, removed even on failure."""
    saved = []

    def hook(module, inputs, h):
        if capture:
            saved.append(h[:, positions].detach())
        if delta is not None:
            h = h.clone()
            h[:, positions] = h[:, positions] + delta
        return h

    handle = model.blocks[depth - 1].register_forward_hook(hook)
    try:
        logits = model(edges)
    finally:
        handle.remove()
    return (logits, saved[0]) if capture else logits


@torch.no_grad()
def fit_directions(model, bank, depth, positions, batch_size=2048):
    """Exact least-squares optimum v_k = mean(h_on - h_off).

    This minimizes sum_i ||h_off_i + v_k - h_on_i||² over every coordinate
    of a single, context-independent additive vector. Its amplitude is one
    natural latent change; no labels, downstream loss, or test data are fitted.
    At several positions the vector is the flattened residual slice.
    Accumulate in float64 to make the result independent of chunk size.
    """
    directions = []
    for off, on in zip(bank['off'], bank['on']):
        total = None
        for a, b in zip(off.split(batch_size), on.split(batch_size)):
            _, ha = forward_at(model, a, depth, positions, capture=True)
            _, hb = forward_at(model, b, depth, positions, capture=True)
            s = (hb.double() - ha.double()).sum(0)
            total = s if total is None else total + s
        directions.append((total / len(off)).float())
    return torch.stack(directions)


def fidelity(delta, target):
    """Return raw fidelity and delta-method SE across independent pair draws.

    Raw can be negative; zero means no effect, one means an exact teacher match.
    The display version floors raw at zero only AFTER averaging the errors.
    """
    error = (delta.double() - target.double()).square().sum(-1)
    signal = target.double().square().sum(-1)
    if signal.sum() == 0:
        raise ValueError("teacher effect is identically zero in this bank")
    ratio = error.mean() / signal.mean()
    influence = (error - ratio * signal) / signal.mean()
    se = influence.std(unbiased=True) / len(error) ** 0.5
    return float(1 - ratio), float(se)


@torch.no_grad()
def measure(model, bank, vectors, depth, positions, batch_size=2048):
    out = {k: [] for k in ('gain_raw', 'steer_raw', 'patch_raw', 'random_raw',
                           'gain_se', 'steer_se', 'accuracy', 'effect_fraction')}
    # A norm-matched permutation control, fixed across checkpoints.
    rng = torch.Generator(device=vectors.device).manual_seed(991)
    random = torch.randn(vectors.shape, generator=rng, device=vectors.device)
    random *= (vectors.flatten(1).norm(dim=1) /
               random.flatten(1).norm(dim=1)).view(-1, 1, 1)
    for j, (off, on) in enumerate(zip(bank['off'], bank['on'])):
        probs = {k: [] for k in ('off', 'on', 'steer', 'patch', 'random')}
        for a, b in zip(off.split(batch_size), on.split(batch_size)):
            la, ha = forward_at(model, a, depth, positions, capture=True)
            lb, hb = forward_at(model, b, depth, positions, capture=True)
            probs['off'].append(la.float().softmax(-1))
            probs['on'].append(lb.float().softmax(-1))
            for key, v in [('steer', vectors[j]), ('patch', hb - ha), ('random', random[j])]:
                logits = forward_at(model, a, depth, positions, delta=v)
                probs[key].append(logits.float().softmax(-1))
        probs = {k: torch.cat(v) for k, v in probs.items()}
        target = (F.one_hot(bank['y_on'][j], probs['on'].shape[-1]) -
                  F.one_hot(bank['y_off'][j], probs['on'].shape[-1])).float()
        for key, pred in [('gain', 'on'), ('steer', 'steer'), ('patch', 'patch'), ('random', 'random')]:
            raw, se = fidelity(probs[pred] - probs['off'], target)
            out[key + '_raw'].append(raw)
            if key in ('gain', 'steer'):
                out[key + '_se'].append(se)
        out['accuracy'].append(float((probs['on'].argmax(-1) == bank['y_on'][j]).float().mean()))
        out['effect_fraction'].append(float((bank['y_on'][j] != bank['y_off'][j]).float().mean()))
    out['vector_norm'] = vectors.flatten(1).norm(dim=1).tolist()
    return out
