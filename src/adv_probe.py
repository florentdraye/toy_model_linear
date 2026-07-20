"""Adversarial linear probes via gradient reversal.

At every (depth, position, intermediate graph layer), a linear classifier tries
to recover the latent node id from the residual stream. The gradient w.r.t. the
hidden state is *reversed* (sign flipped) before flowing back, so the model is
pushed to make h less linearly decodable while the probes themselves are
trained jointly to be as good as possible at recovery.

Only INTERMEDIATE graph layers are attacked (layers 1 .. n_layers-2); the final
answer layer is left alone so the model can still solve the task.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam: float):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class AdversarialProbes(nn.Module):
    """One Linear(d_model, n_classes_per_layer[ell]) probe per (d, p, ell)."""

    def __init__(self, d_model: int, n_blocks: int, seq_len: int,
                 nodes_per_layer: List[int], n_layers: int):
        super().__init__()
        attack_layers = [ell for ell in range(1, n_layers - 1) if nodes_per_layer[ell] > 1]
        self.attack_layers = attack_layers
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.seq_len = seq_len
        # Flat ModuleList; index = (d * seq_len + p) * len(attack_layers) + idx_of_ell
        probes = []
        for d in range(n_blocks + 1):
            for p in range(seq_len):
                for ell in attack_layers:
                    probes.append(nn.Linear(d_model, nodes_per_layer[ell]))
        self.probes = nn.ModuleList(probes)
        self.nodes_per_layer = nodes_per_layer

    def _index(self, d: int, p: int, ell_idx: int) -> int:
        return (d * self.seq_len + p) * len(self.attack_layers) + ell_idx

    def loss(self, hiddens: List[torch.Tensor], nodes_batch: torch.Tensor,
             lam: float) -> torch.Tensor:
        """Returns mean cross-entropy across all (d, p, ell) probes."""
        device = hiddens[0].device
        total = torch.zeros((), device=device)
        count = 0
        for d, h in enumerate(hiddens):                       # h: (B, L, d_model)
            for p in range(self.seq_len):
                h_dp = h[:, p, :]
                h_dp_rev = grad_reverse(h_dp, lam)
                for ell_idx, ell in enumerate(self.attack_layers):
                    probe = self.probes[self._index(d, p, ell_idx)]
                    logits = probe(h_dp_rev)
                    total = total + F.cross_entropy(logits, nodes_batch[:, ell])
                    count += 1
        return total / count
