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

    Selection is uniform over prefix pairs and suffixes. Do not discard pairs
    whose endpoints coincide: their zero teacher effect penalizes leakage.
    """
    rng = torch.Generator().manual_seed(seed)
    edges, nodes = paths['edge_seqs'], paths['nodes']
    # Enumeration is lexicographic; a zero suffix identifies each prefix node.
    suffix_count = radix ** (edges.shape[1] - layer)
    prefix_nodes = nodes[::suffix_count, layer]
    by_node = {int(k): torch.where(prefix_nodes == k)[0] for k in [reference, *targets]}
    pairs = []
    for k in targets:
        # Enumerate the small conditional support exactly. Some reachable nodes
        # have only two prefixes: rejection cannot manufacture more unique test
        # pairs. Cap all latents to the same available coverage, without replacing
        # a difficult latent or duplicating pairs to inflate sample counts.
        a, b = by_node[reference], by_node[k]
        suffix = torch.arange(suffix_count)
        shape = (len(a), len(b), suffix_count)
        ia = (a[:, None, None] * suffix_count + suffix).expand(shape)
        ib = (b[None, :, None] * suffix_count + suffix).expand(shape)
        good = allowed[ia] & allowed[ib]
        ids = torch.stack([ia[good], ib[good]], -1)
        if len(ids) < 2:
            raise ValueError(f'insufficient matched support for latent {k}')
        pairs.append(ids[torch.randperm(len(ids), generator=rng)[:n]])
    coverage = min(len(p) for p in pairs)
    if coverage < n:
        print(f'[pairs] requested {n}; using {coverage} unique pairs per latent (finite support)', flush=True)
    ids = torch.stack([p[:coverage] for p in pairs])
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


def target_fidelity(probabilities, labels):
    """Multiclass Brier skill against the a-priori uniform prediction.

    Unlike a paired-effect fidelity, learning the common reference alone cannot
    create half a unit of apparent target generalization. Uniform prediction is
    exactly zero and a perfect target prediction is exactly one.
    """
    target = F.one_hot(labels, probabilities.shape[-1]).double()
    errors = (probabilities.double() - target).square().sum(-1)
    null_error = 1 - 1 / probabilities.shape[-1]
    return (float(1 - errors.mean() / null_error),
            float(errors.std(unbiased=True) / len(errors) ** .5 / null_error))


def from_hidden(model, h, depth):
    """Continue the existing transformer from a post-block residual state."""
    for block in model.blocks[depth:]:
        h = block(h)
    last = model.ln_f(h)[:, -1]
    if model.frozen_mlp is not None:
        last = model.frozen_mlp(last)
    return model.head(last)


def optimize_directions(model, bank, means, depth, positions, steps=150,
                        lr=.03, batch_size=128):
    """Fit constant causal vectors with frozen weights and held-out validation.

    One vector per target, initialized at the mean shift and constrained to the
    RMS norm of a natural paired hidden change. The final quarter of calibration
    pairs selects iterates; no final-test paths or future models are consulted.
    This is a numerical optimum, not a global-optimality guarantee. It measures
    supervised controllability and is reported beside the unsupervised mean edit.
    """
    device = means.device
    k, n = bank['off'].shape[:2]
    full_positions = list(range(model.cfg.seq_len)) if hasattr(model, 'cfg') else list(range(bank['off'].shape[-1]))
    hs, radii = [], []
    with torch.no_grad():
        for off, on in zip(bank['off'], bank['on']):
            hh, norms = [], []
            for a, b in zip(off.split(1024), on.split(1024)):
                _, ha = forward_at(model, a, depth, full_positions, capture=True)
                _, hb = forward_at(model, b, depth, full_positions, capture=True)
                hh.append(ha)
                norms.append((hb[:, positions].float()-ha[:, positions].float()).flatten(1).square().sum(1))
            hs.append(torch.cat(hh))
            radii.append(torch.cat(norms).mean().sqrt())
        hidden = torch.stack(hs)
        radius = torch.stack(radii).clamp_min(1e-8).view(k, 1, 1)
    n_train = max(1, int(n * .75))
    if n_train == n:
        raise ValueError('vector calibration needs a validation pair')
    unit = torch.nn.Parameter(means / radius)
    optimizer = torch.optim.Adam([unit], lr=lr)
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device=device).manual_seed(817)
    row = torch.arange(k, device=device)[:, None]

    def loss_at(indices):
        h = hidden[row, indices].clone()
        h[:, :, positions] += (unit * radius)[:, None].to(h.dtype)
        probabilities = from_hidden(model, h.flatten(0, 1), depth).float().softmax(-1)
        labels = bank['y_on'][row, indices].flatten()
        targets = F.one_hot(labels, probabilities.shape[-1]).float()
        return (probabilities-targets).square().sum(-1).reshape(k, -1).mean(1)

    validation = torch.arange(n_train, n, device=device)[None].expand(k, -1)
    best, best_step = unit.detach().clone(), torch.zeros(k, dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            initial_loss = loss_at(validation)
            best_loss = initial_loss.clone()
        for step in range(1, steps + 1):
            indices = torch.randint(n_train, (k, min(batch_size, n_train)), generator=gen, device=device)
            loss = loss_at(indices).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                unit.div_(unit.flatten(1).norm(dim=1).clamp_min(1).view(k, 1, 1))
                if step % 10 == 0 or step == steps:
                    val = loss_at(validation)
                    improved = val < best_loss
                    best[improved] = unit.detach()[improved]
                    best_loss[improved] = val[improved]
                    best_step[improved] = step
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
    return best * radius, {'direction_radius': radius.flatten().tolist(),
                          'direction_validation_before': initial_loss.tolist(),
                          'direction_validation_after': best_loss.tolist(),
                          'direction_best_step': best_step.tolist()}


@torch.no_grad()
def measure(model, bank, vectors, depth, positions, batch_size=2048, means=None):
    out = {k: [] for k in ('gain_raw', 'steer_raw', 'patch_raw', 'random_raw',
                           'gain_se', 'steer_se', 'accuracy', 'reference_accuracy',
                           'effect_fraction', 'gain_effect_raw', 'steer_effect_raw', 'mean_raw')}
    # A norm-matched permutation control, fixed across checkpoints.
    rng = torch.Generator(device=vectors.device).manual_seed(991)
    random = torch.randn(vectors.shape, generator=rng, device=vectors.device)
    random *= (vectors.flatten(1).norm(dim=1) /
               random.flatten(1).norm(dim=1)).view(-1, 1, 1)
    for j, (off, on) in enumerate(zip(bank['off'], bank['on'])):
        probs = {k: [] for k in ('off', 'on', 'steer', 'patch', 'random', 'mean')}
        for a, b in zip(off.split(batch_size), on.split(batch_size)):
            la, ha = forward_at(model, a, depth, positions, capture=True)
            lb, hb = forward_at(model, b, depth, positions, capture=True)
            probs['off'].append(la.float().softmax(-1))
            probs['on'].append(lb.float().softmax(-1))
            for key, v in [('steer', vectors[j]), ('patch', hb - ha), ('random', random[j]),
                           ('mean', vectors[j] if means is None else means[j])]:
                logits = forward_at(model, a, depth, positions, delta=v)
                probs[key].append(logits.float().softmax(-1))
        probs = {k: torch.cat(v) for k, v in probs.items()}
        target = (F.one_hot(bank['y_on'][j], probs['on'].shape[-1]) -
                  F.one_hot(bank['y_off'][j], probs['on'].shape[-1])).float()
        for key, pred in [('gain', 'on'), ('steer', 'steer'), ('patch', 'patch'), ('random', 'random'), ('mean', 'mean')]:
            raw, se = target_fidelity(probs[pred], bank['y_on'][j])
            out[key + '_raw'].append(raw)
            if key in ('gain', 'steer'):
                out[key + '_se'].append(se)
                effect_raw, _ = fidelity(probs[pred] - probs['off'], target)
                out[key + '_effect_raw'].append(effect_raw)
        out['accuracy'].append(float((probs['on'].argmax(-1) == bank['y_on'][j]).float().mean()))
        out['reference_accuracy'].append(float((probs['off'].argmax(-1) == bank['y_off'][j]).float().mean()))
        out['effect_fraction'].append(float((bank['y_on'][j] != bank['y_off'][j]).float().mean()))
    out['vector_norm'] = vectors.flatten(1).norm(dim=1).tolist()
    return out
