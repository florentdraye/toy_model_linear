"""Functional encoder-boundary transfer measurements.

The boundary is the complete residual state after a chosen transformer block.
Only parameters upstream of that boundary move in virtual SGD updates; the
downstream transformer and task head stay frozen.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.emergence import from_hidden


def encoder_parameters(model, depth: int = 1):
    if depth != 1:
        raise NotImplementedError("the current experiment uses the post-block-1 boundary")
    keep = ("pos_emb", "tok_emb.", "blocks.0.")
    return OrderedDict((name, value.detach()) for name, value in model.named_parameters()
                       if name == keep[0] or name.startswith(keep[1:]))


def encode(model, params, edges, depth: int = 1):
    """Return the full [batch, sequence, width] post-block residual."""
    if depth != 1:
        raise NotImplementedError("the current experiment uses depth 1")
    h = F.embedding(edges, params["tok_emb.weight"]) + params["pos_emb"]
    h = F.dropout(h, model.drop.p, training=False)
    block = {name.removeprefix("blocks.0."): value for name, value in params.items()
             if name.startswith("blocks.0.")}
    with sdpa_kernel(SDPBackend.MATH):
        return functional_call(model.blocks[0], block, (h,))


def task_loss(model, params, edges, labels, depth: int = 1):
    logits = from_hidden(model, encode(model, params, edges, depth), depth)
    return F.cross_entropy(logits.float(), labels, reduction="mean")


def source_gradient(model, params, edges, labels, depth: int = 1):
    return grad(lambda p: task_loss(model, p, edges, labels, depth))(params)


def scale_tree(tree, scale):
    return OrderedDict((name, scale * value) for name, value in tree.items())


def subtract_step(params, gradient, eta):
    return OrderedDict((name, value - eta * gradient[name]) for name, value in params.items())


def evaluation_state(model, params, edges, labels, depth: int = 1):
    """Return loss, boundary gradient g_i, and the unmodified boundary state."""
    h = encode(model, params, edges, depth).detach().requires_grad_(True)
    logits = from_hidden(model, h, depth)
    losses = F.cross_entropy(logits.float(), labels, reduction="none")
    g = torch.autograd.grad(losses.sum(), h)[0]
    return losses.detach(), g.detach(), h.detach()


def transfer(model, params, eval_edges, parameter_direction, depth: int = 1):
    """J_i direction at the full residual boundary."""
    _, tangent = jvp(lambda p: encode(model, p, eval_edges, depth),
                     (params,), (parameter_direction,))
    return tangent.detach()


def cosine_rows(a, b, eps=1e-20):
    a, b = a.flatten(1).double(), b.flatten(1).double()
    dot = (a * b).sum(1)
    an, bn = a.square().sum(1).sqrt(), b.square().sum(1).sqrt()
    valid = (an * bn) > eps
    cosine = torch.full_like(dot, torch.nan)
    cosine[valid] = dot[valid] / (an[valid] * bn[valid])
    return cosine, dot, an, bn
