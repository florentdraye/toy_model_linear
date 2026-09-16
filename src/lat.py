"""LAT axes from symmetrized contrastive differences, without a reference class."""
import torch


@torch.no_grad()
def principal_axes(differences, seed=9217, tolerance=2e-5, max_iterations=192):
    """Top eigenvector of E[delta delta^T], equivalent to PCA on {+delta,-delta}.

    Input [groups, pairs, width]. Block power iteration uses four trial vectors;
    unconverged groups fall back to full symmetric eigendecomposition. Zero
    contrast groups return a zero axis. No temporal averaging or eigenvector
    initialization from other checkpoints is used.
    """
    if differences.ndim != 3 or not torch.isfinite(differences).all():
        raise ValueError('need finite [groups, pairs, width] contrasts')
    n = differences.shape[-2]
    covariance = differences.transpose(-1, -2) @ differences / n
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    active = trace > 0
    c = covariance / trace.clamp_min(torch.finfo(covariance.dtype).tiny)[:, None, None]
    b, width, _ = c.shape
    rank = min(4, width)
    generator = torch.Generator(device=c.device).manual_seed(seed)
    q = torch.randn(b, width, rank, device=c.device, dtype=c.dtype, generator=generator)
    q = torch.linalg.qr(q).Q
    residual = torch.full_like(trace, float('inf'))
    for iteration in range(max_iterations):
        q = torch.linalg.qr(c @ q).Q
        if (iteration + 1) % 16:
            continue
        cq = c @ q
        small = q.transpose(-1, -2) @ cq
        values, rotation = torch.linalg.eigh((small + small.transpose(-1, -2)) * .5)
        axis = (q @ rotation[..., -1:]).squeeze(-1)
        top = values[..., -1]
        residual = ((c @ axis.unsqueeze(-1)).squeeze(-1) - top[:, None] * axis).norm(dim=-1)
        if iteration >= 47 and (residual[active] <= tolerance).all():
            break
    fallback = active & (residual > tolerance)
    if fallback.any():
        values, eig = torch.linalg.eigh(c[fallback])
        axis[fallback] = eig[..., -1]
        top[fallback] = values[..., -1]
    axis[~active] = 0
    top[~active] = 0
    paired_mean = differences.mean(-2)
    sign = torch.where((axis * paired_mean).sum(-1) < 0, -1., 1.)
    axis *= sign[:, None]
    residual = ((c @ axis.unsqueeze(-1)).squeeze(-1) - top[:, None] * axis).norm(dim=-1)
    if (residual[active] > 2 * tolerance).any():
        raise ValueError('principal-axis eigen residual too large')
    # Check top-eigenvalue identification independently for representative groups.
    check = torch.linspace(0, b - 1, min(6, b), device=c.device).round().long().unique()
    exact = torch.linalg.eigvalsh(c[check])[..., -1]
    if (exact - top[check]).abs().max() > 5e-4:
        raise ValueError('block power iteration missed the leading eigenvalue')
    return axis, dict(explained_energy=top, trace=trace, residual=residual,
                      iterations=iteration + 1, fallback_count=int(fallback.sum()),
                      exact_check_error=float((exact - top[check]).abs().max()))


def balanced_pairs(paths, allowed, layer, classes, per_negative=8, seed=9211, radix=10):
    """Equal count per other class; both paths train-only, identical suffixes.

    Uniform rejection sampling over prefix pairs and suffixes within each
    ordered target/negative class pair. Pair IDs are unique within that pair;
    individual paths can recur across contrasts. Returns [class, pair, off/on].
    """
    if per_negative < 1:
        raise ValueError('need positive per-negative count')
    rng = torch.Generator().manual_seed(seed)
    suffix_count = radix ** (paths['edge_seqs'].shape[1] - layer)
    prefix_nodes = paths['nodes'][::suffix_count, layer]
    prefixes = {k: torch.where(prefix_nodes == k)[0] for k in classes}
    result = []
    for target in classes:
        examples = []
        for negative in classes:
            if target == negative:
                continue
            positive_prefix, negative_prefix = prefixes[target], prefixes[negative]
            selected, seen = [], set()
            for attempt in range(1000):
                size = max(32, per_negative * 2)
                a = negative_prefix[torch.randint(len(negative_prefix), (size,), generator=rng)]
                b = positive_prefix[torch.randint(len(positive_prefix), (size,), generator=rng)]
                suffix = torch.randint(suffix_count, (size,), generator=rng)
                off, on = a * suffix_count + suffix, b * suffix_count + suffix
                good = allowed[off] & allowed[on]
                for pair in zip(off[good].tolist(), on[good].tolist()):
                    if pair not in seen:
                        selected.append(pair); seen.add(pair)
                    if len(selected) == per_negative:
                        break
                if len(selected) == per_negative:
                    break
            if len(selected) != per_negative:
                raise ValueError(f'insufficient training pairs for {negative}->{target}')
            examples.extend(selected)
        result.append(torch.tensor(examples))
    return torch.stack(result)
