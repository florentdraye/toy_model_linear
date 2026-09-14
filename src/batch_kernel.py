"""Exact projected empirical logit-NTK/CE-gradient products and count controls.

g_j = d CE_j / d logits_j; K_ij = J_i J_j^T over all model parameters.
Two projections: correct-endpoint coordinate of K_ij g_j, and g_i^T K_ij g_j.
JVP/VJP implementation avoids materializing per-example parameter Jacobians.
See https://docs.pytorch.org/tutorials/intermediate/neural_tangent_kernels.html
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel


def reference_directions(model, edges, label):
    params = dict(model.named_parameters())
    with sdpa_kernel(SDPBackend.MATH):
        logits = model(edges)[0]
        loss = F.cross_entropy(logits.double()[None], label.reshape(1))
        correct = torch.autograd.grad(logits[label.item()], tuple(params.values()), retain_graph=True)
        loss_grad = torch.autograd.grad(loss, tuple(params.values()))
    directions = [{k: v.detach() for k, v in zip(params, values)} for values in (correct, loss_grad)]
    probabilities = logits.detach().double().softmax(-1)
    g = probabilities.clone()
    g[label.item()] -= 1
    metadata = dict(label=int(label), ce_loss=float(loss.detach()),
                    probabilities=probabilities.cpu().tolist(), g=g.cpu().tolist())
    return directions, metadata


def projected_contributions(model, edges, labels, directions, batch_size=256):
    """Return [example, projection] signed UNNORMALIZED products in FP64."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())
    result = torch.empty(len(edges), len(directions), dtype=torch.float64)
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(edges), batch_size):
            x, y = edges[start:start+batch_size], labels[start:start+batch_size]
            for col, direction in enumerate(directions):
                with torch.no_grad():
                    logits, tangent = jvp(lambda p: functional_call(model, (p, buffers), (x,)),
                                          (params,), (direction,))
                    g = logits.double().softmax(-1)
                    g.scatter_add_(1, y[:, None], -torch.ones(len(y), 1, device=y.device, dtype=torch.float64))
                    values = (g * tangent.double()).sum(-1)
                result[start:start+len(x), col] = values.cpu()
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite kernel products')
    return result


def validate_aggregate(model, ref_edges, ref_label, edges, labels, products):
    """Independent direction: sum training gradients first, then JVP at reference."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with sdpa_kernel(SDPBackend.MATH):
        loss = F.cross_entropy(model(edges).double(), labels, reduction='sum')
        grads = torch.autograd.grad(loss, tuple(params.values()))
        direction = {k: v.detach() for k, v in zip(params, grads)}
        detached = {k: v.detach() for k, v in params.items()}
        with torch.no_grad():
            logits, response = jvp(lambda p: functional_call(model, (p, buffers), (ref_edges,)),
                                   (detached,), (direction,))
            g = logits[0].double().softmax(-1)
            g[ref_label.item()] -= 1
            direct = torch.stack([response[0, ref_label.item()].double(), (g*response[0].double()).sum()]).cpu()
    summed = products.sum(0)
    relative = ((direct-summed).abs() / torch.maximum(direct.abs(), summed.abs()).clamp_min(1e-10))
    torch.testing.assert_close(direct, summed, rtol=3e-3, atol=1e-5)
    return dict(direct=direct.tolist(), sum_per_point=summed.tolist(), relative_error=relative.tolist(),
                full_logit_response=response[0].detach().cpu().tolist(), examples=len(edges))


def count_vector(classes, probabilities, target, target_count, batch_size):
    """Fix all other class counts by deterministic largest-remainder allocation."""
    classes = np.asarray(classes)
    j = int(np.flatnonzero(classes == target)[0])
    if not 0 <= target_count <= batch_size:
        raise ValueError('target count outside batch')
    p = np.array(probabilities, dtype=float, copy=True)
    p[j] = 0
    p /= p.sum()
    expected = (batch_size-target_count)*p
    counts = np.floor(expected).astype(np.int64)
    remaining = batch_size-target_count-int(counts.sum())
    order = np.argsort(-(expected-counts), kind='stable')
    counts[order[:remaining]] += 1
    counts[j] = target_count
    assert counts.sum() == batch_size and counts[j] == target_count
    return counts


def sample_batches(products, pool_classes, classes, probabilities, target, counts,
                   batch_size=16384, repeats=64, seed=8831):
    """products [stage, point, projection]; identical sampled batches across stages.

    Sampling is with replacement within each fixed training pool stratum, as
    in training. Store PCG64 seed and class-count vectors for exact replay.
    """
    products = np.asarray(products, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    strata = [np.flatnonzero(np.asarray(pool_classes) == k) for k in classes]
    if any(len(ix) < 2 for ix in strata):
        raise ValueError('every class needs at least two pool examples')
    vectors = np.stack([count_vector(classes, probabilities, target, n, batch_size) for n in counts])
    result = np.zeros((len(counts), repeats, products.shape[0], products.shape[2]))
    expected = np.zeros((len(counts), products.shape[0], products.shape[2]))
    variance = np.zeros_like(expected)
    for ci, ns in enumerate(vectors):
        for ix, n in zip(strata, ns):
            values = products[:, ix, :]
            expected[ci] += n*values.mean(1)
            variance[ci] += n*values.var(1, ddof=0)
            if n:
                draw = rng.integers(0, len(ix), size=(repeats, int(n)))
                result[ci] += values[:, draw, :].sum(2).transpose(1, 0, 2)
    return dict(responses=result, class_counts=vectors, expected=expected,
                predicted_sd=np.sqrt(variance), target_counts=np.asarray(counts))
