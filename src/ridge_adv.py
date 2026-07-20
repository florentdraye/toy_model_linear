"""Closed-form ridge as a non-linear-decodability regularizer.

For each (residual depth d, sequence position p, intermediate graph layer ell),
build the per-batch optimal ridge predictor of one_hot(node[:, ell]) from the
residual h[d, :, p, :], and add its fit quality to the training loss. The model
then learns residuals on which the best linear probe has high MSE.

Unlike `adv_probe.py`, there are no learnable probe parameters and no gradient
reversal: the "probe" is the closed-form solution to the per-batch ridge
problem, and gradients propagate naturally through torch.linalg.lu_solve.

`ridge_fit_quality` is invariant under X -> k*X (matches probe.py's heuristic of
scaling lam by mean(diag(X^T X))), so the model can't game the penalty by
inflating residual norms.
"""
from typing import List

import torch
import torch.nn.functional as F


def ridge_fit_quality(X: torch.Tensor, Y: torch.Tensor, nu: float) -> torch.Tensor:
    """tr((X^T X + lam_eff I)^{-1} (X^T Y) (X^T Y)^T) / B.

    Equals  1 - MSE(Y, X W*)  when Y is row-one-hot. Returns scalar fp32.
    X (B, D), Y (B, K).
    """
    B, D = X.shape
    XtX = X.T @ X
    diag_mean = XtX.diagonal().mean().detach().clamp(min=1.0)
    lam_eff = nu * diag_mean
    A = XtX + lam_eff * torch.eye(D, device=X.device, dtype=X.dtype)
    M = X.T @ Y                                       # (D, K)
    Z = torch.linalg.solve(A, M)                      # (D, K)
    return (Z * M).sum() / B


def ridge_adv_loss(hiddens: List[torch.Tensor], nodes_batch: torch.Tensor,
                   attack_layers: List[int], nodes_per_layer: List[int],
                   nu: float = 1e-3, tau: float = 0.05) -> torch.Tensor:
    """Soft-max (logsumexp at temperature `tau`) over (d, p, ell) of
    `ridge_fit_quality`. As tau -> 0 this tracks the worst cell, so the model
    can't hide linear info in a few cells while zeroing the mean.

    Returns scalar in roughly [0, 1] + tau * log(n_cells). Computed in fp32
    regardless of autocast dtype on `hiddens`.
    """
    # Exclude the post-final-block residual: the linear classification head reads
    # from it, so any linear scaffolding needed for the answer must live there.
    # We only attack depths 0..n_blocks-1.
    attacked_hiddens = hiddens[:-1]
    L = hiddens[0].shape[1]
    cells = []
    for h in attacked_hiddens:
        h32 = h.float()
        for p in range(L):
            X = h32[:, p, :]
            B, D = X.shape
            XtX = X.T @ X
            diag_mean = XtX.diagonal().mean().detach().clamp(min=1.0)
            lam_eff = nu * diag_mean
            A = XtX + lam_eff * torch.eye(D, device=X.device, dtype=X.dtype)
            LU, pivots = torch.linalg.lu_factor(A)
            for ell in attack_layers:
                K = nodes_per_layer[ell]
                Y = F.one_hot(nodes_batch[:, ell].long(), num_classes=K).float()
                M = X.T @ Y
                Z = torch.linalg.lu_solve(LU, pivots, M)
                cells.append((Z * M).sum() / B)
    if not cells:
        return torch.zeros((), device=hiddens[0].device, dtype=torch.float32)
    stacked = torch.stack(cells)
    return tau * torch.logsumexp(stacked / tau, dim=0)
