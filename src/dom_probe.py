"""Difference-of-means probes for intermediate graph latents, tracked over training.

For each intermediate graph layer ell, sample K node ids from the reachable
support. On the train split, the DoM direction for node c is:

    w_{ell,c} = mean(X | node_ell == c) - mean(X | node_ell != c)

At each (transformer depth d, position p), scoring test examples with w gives
one scalar per (example, c); we report AUC on the held-out set. Sampled node
ids and train/test indices are frozen at setup so numbers are comparable
across time.

One tracker call = one forward pass on the full split + K matmuls per (d, p).
Cheap enough to run every ~10 training steps for the toy sizes here.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from .model import ToyTransformer


@dataclass
class DoMProbeConfig:
    n_nodes: int = 20
    every_steps: int = 0            # 0 disables
    eval_batch: int = 16384
    sample_seed: int = 0
    probe_max_train: int = 60000    # cap train examples used for probe (0 = all)
    probe_max_test: int = 5000      # cap test examples used for probe (0 = all)
    # Steering intervention: at each probe step, at each layer's currently-best
    # (d, p) DoM cell, add alpha * (mu+ - mu-) to the residual and measure how
    # often the model's output prediction lands on a graph-6 node reachable
    # from the target class. 0 disables.
    steer_n_classes: int = 0        # >0 to enable steering; K target classes per layer
    steer_alpha: float = 1.0
    # 1-hidden-layer MLP probes at every (d, p) cell, fit from scratch each
    # probe step. Batched across cells so cost is small. mlp_hidden = 0 disables.
    mlp_hidden: int = 0
    mlp_iters: int = 100
    mlp_lr: float = 3e-2
    mlp_max_train: int = 5000
    mlp_activation: str = "relu"    # "relu" | "gelu"
    # Multinomial logistic regression probes at every (d, p) cell, fit from
    # scratch each probe step. Batched across cells. 0 iters disables.
    logreg_iters: int = 0
    logreg_lr: float = 3e-2
    logreg_max_train: int = 5000
    # Probe-batch gradient snapshots. At each probe step, take one virtual
    # SGD step on *fixed* per-class batches at each positive fraction and
    # record how h moves on the test subset.
    # Denoises the natural-batch deriv: the fixed batches guarantee the
    # gradient always carries signal for the latent being probed.
    probe_grad_batch_size: int = 200
    probe_grad_active_fracs: tuple[float, ...] = (0.01, 0.1, 0.5)
    probe_grad_lr: float = 3e-3
    ntk_pair_group_size: int = 8
    ntk_pair_layers: tuple[int, ...] = (3, 4)
    lr_direction_iters: int = 2000
    lr_direction_lr: float = 0.05


def _fit_batched_mlp(X_tr: torch.Tensor, y_tr: torch.Tensor,
                     X_te: torch.Tensor, y_te: torch.Tensor,
                     hidden: int, n_iters: int, lr: float,
                     activation: str, n_classes: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit M independent 1-hidden-layer MLP probes in parallel via batched matmul.

    X_tr: (M, Ntr, d). X_te: (M, Nte, d). y_tr, y_te: (Ntr,), (Nte,) shared.
    Returns (top1, top5) each of shape (M,).
    """
    M, Ntr, d = X_tr.shape
    C = n_classes
    device = X_tr.device
    X_tr = X_tr.detach()
    X_te = X_te.detach()
    with torch.enable_grad():
        W1 = torch.empty(M, d, hidden, device=device).normal_(0, (2.0 / d) ** 0.5)
        b1 = torch.zeros(M, hidden, device=device)
        W2 = torch.empty(M, hidden, C, device=device).normal_(0, (1.0 / hidden) ** 0.5)
        b2 = torch.zeros(M, C, device=device)
        for t in (W1, b1, W2, b2):
            t.requires_grad_(True)
        act = torch.relu if activation == "relu" else torch.nn.functional.gelu
        optim = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
        y_tr_flat = y_tr.unsqueeze(0).expand(M, -1).reshape(-1)   # (M*Ntr,)
        for _ in range(n_iters):
            hh = act(torch.bmm(X_tr, W1) + b1.unsqueeze(1))       # (M, Ntr, hidden)
            logits = torch.bmm(hh, W2) + b2.unsqueeze(1)          # (M, Ntr, C)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, C), y_tr_flat, reduction="mean")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    with torch.no_grad():
        hh_te = act(torch.bmm(X_te, W1) + b1.unsqueeze(1))
        logits_te = torch.bmm(hh_te, W2) + b2.unsqueeze(1)    # (M, Nte, C)
        pred1 = logits_te.argmax(dim=-1)
        top1 = (pred1 == y_te.unsqueeze(0)).float().mean(dim=1).cpu()
        k5 = min(5, C)
        top5_pred = logits_te.topk(k5, dim=-1).indices        # (M, Nte, 5)
        top5 = (top5_pred == y_te.view(1, -1, 1)).any(dim=-1).float().mean(dim=1).cpu()
    return top1, top5


def _fit_batched_logreg(X_tr: torch.Tensor, y_tr: torch.Tensor,
                        X_te: torch.Tensor, y_te: torch.Tensor,
                        n_iters: int, lr: float, n_classes: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit M multinomial logistic regressions in parallel via bmm.
    X_tr: (M, Ntr, d). Returns (top1, top5) each (M,)."""
    M, Ntr, d = X_tr.shape
    C = n_classes
    device = X_tr.device
    X_tr = X_tr.detach(); X_te = X_te.detach()
    with torch.enable_grad():
        W = torch.zeros(M, d, C, device=device, requires_grad=True)
        b = torch.zeros(M, C, device=device, requires_grad=True)
        optim = torch.optim.Adam([W, b], lr=lr)
        y_tr_flat = y_tr.unsqueeze(0).expand(M, -1).reshape(-1)
        for _ in range(n_iters):
            logits = torch.bmm(X_tr, W) + b.unsqueeze(1)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, C), y_tr_flat, reduction="mean")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    with torch.no_grad():
        logits_te = torch.bmm(X_te, W) + b.unsqueeze(1)   # (M, Nte, C)
        pred1 = logits_te.argmax(dim=-1)
        top1 = (pred1 == y_te.unsqueeze(0)).float().mean(dim=1).cpu()
        k5 = min(5, C)
        top5_pred = logits_te.topk(k5, dim=-1).indices
        top5 = (top5_pred == y_te.view(1, -1, 1)).any(dim=-1).float().mean(dim=1).cpu()
    return top1, top5


@torch.no_grad()
def _forward_with_steer(model: ToyTransformer, edges: torch.Tensor,
                        vec_per_row: torch.Tensor, at_d: int, at_p: int
                        ) -> torch.Tensor:
    """Forward pass with vec_per_row added to residual at (depth at_d, position at_p).

    vec_per_row: (N, d_model). at_d: 0 = raw embedding; k>0 = after block k.
    Returns logits (N, n_classes). Autocast bf16, matches model.forward().
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.tok_emb(edges) + model.pos_emb
        h = model.drop(h)
        if at_d == 0:
            h[:, at_p] = h[:, at_p] + vec_per_row.to(h.dtype)
        for i, block in enumerate(model.blocks):
            h = block(h)
            if at_d == i + 1:
                h[:, at_p] = h[:, at_p] + vec_per_row.to(h.dtype)
        h_final = model.ln_f(h)
        last = h_final[:, -1]
        if model.frozen_mlp is not None:
            last = model.frozen_mlp(last)
        return model.head(last)


@torch.no_grad()
def _binary_auc_columns(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUC per column via Mann-Whitney U. scores, labels: (N, K). Returns (K,)."""
    N, K = scores.shape
    n_pos = labels.sum(0)
    n_neg = N - n_pos
    idx = scores.argsort(dim=0)                       # (N, K)
    arange = torch.arange(1, N + 1, device=scores.device, dtype=torch.float64
                          ).unsqueeze(1).expand(-1, K)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks.scatter_(0, idx, arange)
    pos_rank_sum = (ranks * labels.double()).sum(0)
    n_pos_d, n_neg_d = n_pos.double(), n_neg.double()
    valid = (n_pos > 0) & (n_neg > 0)
    auc_valid = (pos_rank_sum - n_pos_d * (n_pos_d + 1) / 2) / (n_pos_d * n_neg_d)
    return torch.where(valid, auc_valid, torch.full_like(auc_valid, float("nan"))).float()


class DoMProbeTracker:
    """Runs DoM probes on a fixed eval set at intervals during training."""

    def __init__(self,
                 edges: torch.Tensor, nodes: torch.Tensor,
                 train_idx: torch.Tensor, test_idx: torch.Tensor,
                 n_layers: int, nodes_per_layer: List[int],
                 cfg: DoMProbeConfig):
        self.cfg = cfg
        self.device = edges.device

        # Subsample the probe eval set to keep the hidden cube small.
        rng = torch.Generator(device="cpu").manual_seed(cfg.sample_seed)
        if cfg.probe_max_train > 0 and cfg.probe_max_train < train_idx.numel():
            perm = torch.randperm(train_idx.numel(), generator=rng)[:cfg.probe_max_train]
            train_idx = train_idx[perm.to(train_idx.device)]
        if cfg.probe_max_test > 0 and cfg.probe_max_test < test_idx.numel():
            perm = torch.randperm(test_idx.numel(), generator=rng)[:cfg.probe_max_test]
            test_idx = test_idx[perm.to(test_idx.device)]

        n_tr, n_te = train_idx.numel(), test_idx.numel()
        # Pack the probe subset contiguously and re-index into it.
        self.edges = torch.cat([edges[train_idx], edges[test_idx]], dim=0)
        nodes = torch.cat([nodes[train_idx], nodes[test_idx]], dim=0)
        self.train_idx = torch.arange(n_tr, device=self.device)
        self.test_idx = torch.arange(n_tr, n_tr + n_te, device=self.device)
        train_idx, test_idx = self.train_idx, self.test_idx   # for setup below

        self.target_layers = [ell for ell in range(n_layers)
                              if nodes_per_layer[ell] > 1]

        self.chosen: Dict[int, torch.Tensor] = {}
        for ell in self.target_layers:
            reachable = nodes[:, ell].unique()
            if reachable.numel() <= cfg.n_nodes:
                self.chosen[ell] = reachable
            else:
                perm = torch.randperm(reachable.numel(), generator=rng).to(self.device)
                self.chosen[ell] = reachable[perm[:cfg.n_nodes]]

        self.Y_train: Dict[int, torch.Tensor] = {}
        self.Y_test: Dict[int, torch.Tensor] = {}
        self.n_pos_tr: Dict[int, torch.Tensor] = {}
        self.n_neg_tr: Dict[int, torch.Tensor] = {}
        # Multi-class targets: full label vectors + reachable class ids.
        self.y_train_full: Dict[int, torch.Tensor] = {}
        self.y_test_full: Dict[int, torch.Tensor] = {}
        self.reachable_classes: Dict[int, torch.Tensor] = {}
        self.Y_train_mc: Dict[int, torch.Tensor] = {}      # (Ntr, C_ell) one-hot on reachable
        self.n_pos_mc: Dict[int, torch.Tensor] = {}
        self.n_neg_mc: Dict[int, torch.Tensor] = {}
        for ell in self.target_layers:
            c = self.chosen[ell]
            mid_tr = nodes[train_idx, ell]
            mid_te = nodes[test_idx, ell]
            Y_tr = (mid_tr.unsqueeze(1) == c.unsqueeze(0)).float()
            Y_te = (mid_te.unsqueeze(1) == c.unsqueeze(0)).float()
            self.Y_train[ell] = Y_tr
            self.Y_test[ell] = Y_te
            self.n_pos_tr[ell] = Y_tr.sum(0).clamp(min=1.0)
            self.n_neg_tr[ell] = (Y_tr.shape[0] - Y_tr.sum(0)).clamp(min=1.0)

            reachable = nodes[:, ell].unique()
            self.reachable_classes[ell] = reachable
            # Remap raw node ids -> [0, C_ell). y_full stores remapped labels.
            remap = torch.full((int(reachable.max().item()) + 1,), -1,
                               dtype=torch.long, device=self.device)
            remap[reachable] = torch.arange(reachable.numel(), device=self.device)
            self.y_train_full[ell] = remap[mid_tr]
            self.y_test_full[ell] = remap[mid_te]
            Y_tr_mc = torch.nn.functional.one_hot(
                self.y_train_full[ell], reachable.numel()).float()
            self.Y_train_mc[ell] = Y_tr_mc
            self.n_pos_mc[ell] = Y_tr_mc.sum(0).clamp(min=1.0)
            self.n_neg_mc[ell] = (Y_tr_mc.shape[0] - Y_tr_mc.sum(0)).clamp(min=1.0)

        # Steering setup: K target classes per layer (fixed) + reachability masks
        # into the graph's final layer (which is what the model's head outputs).
        self.steer_targets: Dict[int, torch.Tensor] = {}
        self.steer_reachable: Dict[int, torch.Tensor] = {}
        self.y_ell: Dict[int, torch.Tensor] = {}    # (Ntr+Nte,) raw layer-ell node id
        # Save raw node table for the probe subset so run() can look up labels.
        self.nodes_sub = nodes
        self.n_out = int(nodes[:, -1].max().item()) + 1
        if cfg.steer_n_classes > 0:
            steer_rng = torch.Generator(device="cpu").manual_seed(cfg.sample_seed + 1)
            for ell in self.target_layers:
                reachable = self.reachable_classes[ell]
                Ks = min(cfg.steer_n_classes, reachable.numel())
                perm = torch.randperm(reachable.numel(), generator=steer_rng)[:Ks]
                targets = reachable[perm.to(self.device)]        # (Ks,)
                self.steer_targets[ell] = targets
                mask = torch.zeros(Ks, self.n_out, device=self.device, dtype=torch.bool)
                for k, c in enumerate(targets.tolist()):
                    outs = nodes[nodes[:, ell] == c, -1].unique()
                    mask[k, outs] = True
                self.steer_reachable[ell] = mask
                self.y_ell[ell] = nodes[:, ell]

        self.history: Dict = {
            "step": [],
            "cubes": {ell: [] for ell in self.target_layers},
            "cubes_diff": {ell: [] for ell in self.target_layers},     # binary DoM AUC on h(t)-h(prev probe)
            "cubes_deriv": {ell: [] for ell in self.target_layers},    # binary DoM AUC on h(t)-h(t-1step)
            "top1": {ell: [] for ell in self.target_layers},
            "top5": {ell: [] for ell in self.target_layers},
            "top1_diff": {ell: [] for ell in self.target_layers},      # multi-class DoM top-1 on h(t)-h(prev)
            "top5_diff": {ell: [] for ell in self.target_layers},
            "top1_deriv": {ell: [] for ell in self.target_layers},     # multi-class DoM top-1 on h(t)-h(t-1)
            "top5_deriv": {ell: [] for ell in self.target_layers},
            "top1_deriv10": {ell: [] for ell in self.target_layers},   # multi-class DoM top-1 on h(t)-h(t-10)
            "top5_deriv10": {ell: [] for ell in self.target_layers},
            "top1_deriv50": {ell: [] for ell in self.target_layers},   # multi-class DoM top-1 on h(t)-h(t-50)
            "top5_deriv50": {ell: [] for ell in self.target_layers},
            "top1_loss_grad": {ell: [] for ell in self.target_layers}, # multi-class DoM top-1 on -dL/dh
            "top5_loss_grad": {ell: [] for ell in self.target_layers},
            "steer": {ell: [] for ell in self.target_layers},          # (Ks,) per step
            "steer_baseline": {ell: [] for ell in self.target_layers}, # (Ks,) per step
            "steer_cell": {ell: [] for ell in self.target_layers},     # (2,) [d*, p*] per step
            "mlp_top1": {ell: [] for ell in self.target_layers},       # (D, L) per step
            "mlp_top5": {ell: [] for ell in self.target_layers},
            "logreg_top1": {ell: [] for ell in self.target_layers},    # (D, L) per step
            "logreg_top5": {ell: [] for ell in self.target_layers},
        }
        # Caches for over-time diff and derivative lines:
        #   h_prev_probe: h at the previous probe step, for diff = h(t) - h(prev)
        #   h_pending:    h one real training step before the next probe step,
        #                 to compute deriv = h(t) - h(t-1) at the probe step.
        self.h_prev_probe: torch.Tensor | None = None
        self.h_pending: torch.Tensor | None = None
        self.pending_deriv_step: int = -1
        self.h_pending10: torch.Tensor | None = None
        self.pending_deriv10_step: int = -1
        self.h_pending50: torch.Tensor | None = None
        self.pending_deriv50_step: int = -1
        # CPU caches of hidden-state snapshots on the test subset, one per
        # probe step. Consumed at end-of-training by finalize_dot_products()
        # to score cos(w_c^final, X(t)) for X in {h, h-h_prev_probe,
        # h_t-h_{t-1}, h_t-h_{t-10}}.
        self.h_test_history: list[torch.Tensor] = []
        self.deriv_test_history: list[torch.Tensor] = []
        self.deriv_steps: list[int] = []
        self.deriv10_test_history: list[torch.Tensor] = []
        self.deriv10_steps: list[int] = []
        self.deriv50_test_history: list[torch.Tensor] = []
        self.deriv50_steps: list[int] = []
        self.loss_grad_test_history: list[torch.Tensor] = []
        self.jvp_update_test_history: list[torch.Tensor] = []
        self.jvp_update_steps: list[int] = []
        self.ntk_state_history: list[dict[str, torch.Tensor]] = []
        # Filled in by finalize_dot_products(); cosine alignments shaped
        # (T, D, L, K) per target layer.
        self.dot_rep_active: Dict[int, torch.Tensor] = {}
        self.dot_rep_inactive: Dict[int, torch.Tensor] = {}
        self.dot_diff_active: Dict[int, torch.Tensor] = {}
        self.dot_diff_inactive: Dict[int, torch.Tensor] = {}
        self.dot_deriv_active: Dict[int, torch.Tensor] = {}
        self.dot_deriv_inactive: Dict[int, torch.Tensor] = {}
        self.dot_deriv10_active: Dict[int, torch.Tensor] = {}
        self.dot_deriv10_inactive: Dict[int, torch.Tensor] = {}
        self.dot_deriv50_active: Dict[int, torch.Tensor] = {}
        self.dot_deriv50_inactive: Dict[int, torch.Tensor] = {}
        self.dot_loss_grad_active: Dict[int, torch.Tensor] = {}
        self.dot_loss_grad_inactive: Dict[int, torch.Tensor] = {}
        self.dot_loss_grad_active_std: Dict[int, torch.Tensor] = {}
        self.dot_loss_grad_inactive_std: Dict[int, torch.Tensor] = {}
        self.dot_jvp_update_active: Dict[int, torch.Tensor] = {}
        self.dot_jvp_update_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_rep_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_rep_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_diff_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_diff_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_deriv_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_deriv_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_deriv10_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_deriv10_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_deriv50_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_deriv50_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_loss_grad_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_loss_grad_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_loss_grad_active_std: Dict[int, torch.Tensor] = {}
        self.raw_dot_loss_grad_inactive_std: Dict[int, torch.Tensor] = {}
        self.raw_dot_jvp_update_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_jvp_update_inactive: Dict[int, torch.Tensor] = {}
        self.dot_ntk_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.raw_dot_ntk_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.dot_ntk_grad_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.raw_dot_ntk_grad_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.ntk_pair_hist: Dict[str, Dict[int, list[torch.Tensor]]] = {
            key: {ell: [] for ell in self.target_layers}
            for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.ntk_pair_p_hist: Dict[str, Dict[int, list[torch.Tensor]]] = {
            key: {ell: [] for ell in self.target_layers}
            for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.norm_diff_active: Dict[int, torch.Tensor] = {}
        self.norm_diff_inactive: Dict[int, torch.Tensor] = {}
        self.norm_deriv_active: Dict[int, torch.Tensor] = {}
        self.norm_deriv_inactive: Dict[int, torch.Tensor] = {}
        self.norm_deriv10_active: Dict[int, torch.Tensor] = {}
        self.norm_deriv10_inactive: Dict[int, torch.Tensor] = {}
        self.norm_deriv50_active: Dict[int, torch.Tensor] = {}
        self.norm_deriv50_inactive: Dict[int, torch.Tensor] = {}
        self.dot_probe_active: Dict[float, Dict[int, torch.Tensor]] = {
            frac: {} for frac in self.cfg.probe_grad_active_fracs
        }
        self.dot_probe_inactive: Dict[float, Dict[int, torch.Tensor]] = {
            frac: {} for frac in self.cfg.probe_grad_active_fracs
        }
        # (D, L) top1_rep at final step, used by the plotter to pick best (d, p).
        self.final_top1_rep: Dict[int, torch.Tensor] = {}

        # Logistic-regression probe direction, computed only at the DoM-selected
        # cell (d*, p*). All cubes below have shape (T, K).
        self.lr_cell: Dict[int, torch.Tensor] = {}                 # (2,) [d*, p*]
        self.W_lr_final: Dict[int, torch.Tensor] = {}              # (K, d_model)
        self.dot_lr_rep_active: Dict[int, torch.Tensor] = {}
        self.dot_lr_rep_inactive: Dict[int, torch.Tensor] = {}
        self.dot_lr_diff_active: Dict[int, torch.Tensor] = {}
        self.dot_lr_diff_inactive: Dict[int, torch.Tensor] = {}
        self.dot_lr_deriv_active: Dict[int, torch.Tensor] = {}
        self.dot_lr_deriv_inactive: Dict[int, torch.Tensor] = {}
        self.dot_lr_deriv10_active: Dict[int, torch.Tensor] = {}
        self.dot_lr_deriv10_inactive: Dict[int, torch.Tensor] = {}
        self.dot_lr_deriv50_active: Dict[int, torch.Tensor] = {}
        self.dot_lr_deriv50_inactive: Dict[int, torch.Tensor] = {}
        self.dot_lr_loss_grad_active: Dict[int, torch.Tensor] = {}
        self.dot_lr_loss_grad_inactive: Dict[int, torch.Tensor] = {}
        self.dot_lr_loss_grad_active_std: Dict[int, torch.Tensor] = {}
        self.dot_lr_loss_grad_inactive_std: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_rep_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_rep_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_diff_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_diff_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_deriv_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_deriv_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_deriv10_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_deriv10_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_deriv50_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_deriv50_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_loss_grad_active: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_loss_grad_inactive: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_loss_grad_active_std: Dict[int, torch.Tensor] = {}
        self.raw_dot_lr_loss_grad_inactive_std: Dict[int, torch.Tensor] = {}
        self.dot_lr_ntk_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.raw_dot_lr_ntk_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.dot_lr_ntk_grad_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }
        self.raw_dot_lr_ntk_grad_pair: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
        }

        # Full-subset snapshots at the last probe step, used at end of training
        # to fit a single multinomial logistic-regression probe per feature per
        # layer at the DoM-selected cell. None until first probe step.
        self.h_full_last: torch.Tensor = None
        self.diff_full_last: torch.Tensor = None
        self.deriv_full_last: torch.Tensor = None
        self.deriv10_full_last: torch.Tensor = None
        self.deriv50_full_last: torch.Tensor = None
        self.loss_grad_full_last: torch.Tensor = None
        # Curves computed at end of training: for each feature type, fit both a
        # multiclass DoM direction and a multinomial-logreg direction ONCE at
        # the DoM-selected cell using last-probe train features, then evaluate
        # top-1 on every stored test snapshot -> shape (T,) per (feature, ell).
        self.final_dom_top1_curve: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in
            ("rep", "diff", "deriv", "deriv10", "deriv50", "loss_grad")
        }
        self.final_lr_top1_curve: Dict[str, Dict[int, torch.Tensor]] = {
            key: {} for key in
            ("rep", "diff", "deriv", "deriv10", "deriv50", "loss_grad")
        }

        # Fixed per-class probe batches for the virtual-grad analysis.
        # probe_grad_batches[frac][ell] is (K, batch_size) of indices into
        # self.edges, sampled with frac positives per c.
        self.probe_grad_batches: Dict[float, Dict[int, torch.Tensor]] = {
            frac: {} for frac in self.cfg.probe_grad_active_fracs
        }
        self.ntk_pair_indices: Dict[int, Dict[str, torch.Tensor]] = {}
        # Temporary GPU cache set by run() and consumed by _run_probe_grad_step().
        self._h_test_now: torch.Tensor | None = None
        # Per-probe-step CPU history of class-conditioned delta_h means:
        #   probe_delta_active_hist[frac][ell] is a list of (K, D, L, d)
        #   CPU tensors, one per probe step; same for _inactive.
        self.probe_delta_active_hist: Dict[float, Dict[int, list[torch.Tensor]]] = {
            frac: {ell: [] for ell in self.target_layers}
            for frac in self.cfg.probe_grad_active_fracs
        }
        self.probe_delta_inactive_hist: Dict[float, Dict[int, list[torch.Tensor]]] = {
            frac: {ell: [] for ell in self.target_layers}
            for frac in self.cfg.probe_grad_active_fracs
        }
        # Build the fixed probe batches now.
        probe_rng = torch.Generator(device="cpu").manual_seed(cfg.sample_seed + 2)
        train_idx_cpu = self.train_idx.cpu()
        for frac in self.cfg.probe_grad_active_fracs:
            n_pos = int(round(cfg.probe_grad_batch_size * frac))
            n_neg = cfg.probe_grad_batch_size - n_pos
            for ell in self.target_layers:
                c_ids = self.chosen[ell].cpu()                         # (K,)
                K = c_ids.numel()
                batches = torch.empty(K, cfg.probe_grad_batch_size,
                                      dtype=torch.long)
                train_labels_ell = self.nodes_sub[train_idx_cpu, ell].cpu()
                for k_idx in range(K):
                    c = int(c_ids[k_idx].item())
                    pos_mask = (train_labels_ell == c)
                    pos_train = train_idx_cpu[pos_mask]
                    neg_train = train_idx_cpu[~pos_mask]
                    n_pos_actual = min(n_pos, pos_train.numel())
                    n_neg_actual = min(n_neg, neg_train.numel())
                    perm_p = torch.randperm(pos_train.numel(),
                                            generator=probe_rng)[:n_pos_actual]
                    perm_n = torch.randperm(neg_train.numel(),
                                            generator=probe_rng)[:n_neg_actual]
                    picks_p = pos_train[perm_p]
                    picks_n = neg_train[perm_n]
                    picks = torch.cat([picks_p, picks_n], dim=0)
                    if picks.numel() < cfg.probe_grad_batch_size:
                        # Pad by repeating whichever side exists. This keeps
                        # the batch shape fixed for rare classes or tiny fracs.
                        pad_src = picks_n if picks_n.numel() > 0 else picks_p
                        extra = cfg.probe_grad_batch_size - picks.numel()
                        reps = pad_src[torch.arange(extra) % pad_src.numel()]
                        picks = torch.cat([picks, reps], dim=0)
                    shuf = torch.randperm(picks.numel(), generator=probe_rng)
                    batches[k_idx] = picks[shuf]
                self.probe_grad_batches[frac][ell] = batches.to(self.device)

        ntk_rng = torch.Generator(device="cpu").manual_seed(cfg.sample_seed + 3)
        Nte = self.test_idx.numel()
        local_test = torch.arange(Nte, dtype=torch.long)
        S = cfg.ntk_pair_group_size
        for ell in self.target_layers:
            if ell not in cfg.ntk_pair_layers:
                continue
            K = self.Y_test[ell].shape[1]
            active = torch.empty(K, S, dtype=torch.long)
            inactive = torch.empty(K, S, dtype=torch.long)
            y_cpu = self.Y_test[ell].cpu().bool()
            for k_idx in range(K):
                pos = local_test[y_cpu[:, k_idx]]
                neg = local_test[~y_cpu[:, k_idx]]
                for dst, pool in [(active, pos), (inactive, neg)]:
                    if pool.numel() == 0:
                        pool = local_test
                    perm = torch.randperm(pool.numel(), generator=ntk_rng)
                    picks = pool[perm[:min(S, pool.numel())]]
                    if picks.numel() < S:
                        reps = picks[torch.arange(S - picks.numel()) % picks.numel()]
                        picks = torch.cat([picks, reps], dim=0)
                    dst[k_idx] = self.test_idx[picks]
            self.ntk_pair_indices[ell] = {
                "active": active.to(self.device),
                "inactive": inactive.to(self.device),
            }

    @torch.no_grad()
    def _forward_hiddens(self, model: ToyTransformer,
                         edges: torch.Tensor) -> torch.Tensor:
        model.eval()
        N = edges.shape[0]
        out = None
        for i in range(0, N, self.cfg.eval_batch):
            chunk = edges[i:i + self.cfg.eval_batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, h = model(chunk, return_hidden=True)
            hs = torch.stack(h, dim=0).float()
            if out is None:
                D, _, L, d = hs.shape
                out = torch.empty(D, N, L, d, device=self.device, dtype=torch.float32)
            out[:, i:i + chunk.shape[0]] = hs
        return out

    @torch.no_grad()
    def _collect_hiddens(self, model: ToyTransformer) -> torch.Tensor:
        return self._forward_hiddens(model, self.edges)

    def _collect_loss_hgrads(self, model: ToyTransformer,
                             indices: torch.Tensor = None) -> torch.Tensor:
        """Collect -dL/dh on a subset for the usual CE loss.

        indices: (M,) long tensor of positions into self.edges. Defaults to
        self.test_idx. Returns (D, M, L, d_model).
        """
        was_training = model.training
        model.eval()
        if indices is None:
            indices = self.test_idx
        edges = self.edges[indices]
        labels = self.nodes_sub[indices, -1]
        N = edges.shape[0]
        out = None
        for i in range(0, N, self.cfg.eval_batch):
            chunk = edges[i:i + self.cfg.eval_batch]
            y = labels[i:i + self.cfg.eval_batch]
            for p in model.parameters():
                p.grad = None
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, hiddens = model(chunk, return_hidden=True)
                for h in hiddens:
                    h.retain_grad()
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), y, reduction="sum")
            loss.backward()
            grads = torch.stack([-h.grad.detach().float() for h in hiddens], dim=0)
            if out is None:
                D, _, L, d = grads.shape
                out = torch.empty(D, N, L, d, device=self.device, dtype=torch.float32)
            out[:, i:i + chunk.shape[0]] = grads
            for p in model.parameters():
                p.grad = None
        if was_training:
            model.train()
        return out

    def collect_update_jvp(self, model: ToyTransformer,
                           theta_dot: Dict[str, torch.Tensor],
                           step: int) -> None:
        """Collect J_h(theta) @ theta_dot on the fixed test subset.

        theta_dot is the actual optimizer update direction for model
        parameters, keyed by named_parameters(). For plain SGD this is
        -lr * (grad + weight_decay * param), after gradient clipping.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.func import functional_call, jvp

        was_training = model.training
        model.eval()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        tangent = {
            name: theta_dot.get(name, torch.zeros_like(param)).detach()
            for name, param in params.items()
        }

        def hidden_fn(p, edges):
            _, hiddens = functional_call(
                model, (p, buffers), (edges,), {"return_hidden": True}
            )
            return torch.stack(hiddens, dim=0).float()

        edges = self.edges[self.test_idx]
        N = edges.shape[0]
        out = None
        for i in range(0, N, self.cfg.eval_batch):
            chunk = edges[i:i + self.cfg.eval_batch]
            with sdpa_kernel(SDPBackend.MATH), \
                 torch.autocast("cuda", dtype=torch.bfloat16):
                _, dh = jvp(lambda p: hidden_fn(p, chunk), (params,), (tangent,))
            dh = dh.detach().float()
            if out is None:
                D, _, L, d = dh.shape
                out = torch.empty(D, N, L, d, device=self.device, dtype=torch.float32)
            out[:, i:i + chunk.shape[0]] = dh
            del dh
        self.jvp_update_test_history.append(out.cpu())
        self.jvp_update_steps.append(step)
        if was_training:
            model.train()

    def _loss_hgrads_for_edges(self, model: ToyTransformer,
                               edges: torch.Tensor,
                               labels: torch.Tensor) -> torch.Tensor:
        for p in model.parameters():
            p.grad = None
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, hiddens = model(edges, return_hidden=True)
            for h in hiddens:
                h.retain_grad()
            loss = torch.nn.functional.cross_entropy(
                logits.float(), labels, reduction="sum")
        loss.backward()
        grads = torch.stack([-h.grad.detach().float() for h in hiddens], dim=0)
        for p in model.parameters():
            p.grad = None
        return grads

    def _collect_p_hgrads(self, model: ToyTransformer,
                          indices: torch.Tensor = None) -> torch.Tensor:
        """Collect +dP/dh where P = logits[:, y_correct] (correct-class logit).

        Same signature/shape as _collect_loss_hgrads: (D, M, L, d_model).
        Sign is positive: this is the direction that INCREASES the correct-class
        logit, analogous to -dL/dh being the descent direction on loss.
        """
        was_training = model.training
        model.eval()
        if indices is None:
            indices = self.test_idx
        edges = self.edges[indices]
        labels = self.nodes_sub[indices, -1]
        N = edges.shape[0]
        out = None
        for i in range(0, N, self.cfg.eval_batch):
            chunk = edges[i:i + self.cfg.eval_batch]
            y = labels[i:i + self.cfg.eval_batch]
            for p in model.parameters():
                p.grad = None
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, hiddens = model(chunk, return_hidden=True)
                for h in hiddens:
                    h.retain_grad()
                P = logits.float().gather(1, y.unsqueeze(1)).sum()
            P.backward()
            grads = torch.stack([h.grad.detach().float() for h in hiddens], dim=0)
            if out is None:
                D, _, L, d = grads.shape
                out = torch.empty(D, N, L, d, device=self.device, dtype=torch.float32)
            out[:, i:i + chunk.shape[0]] = grads
            for p in model.parameters():
                p.grad = None
        if was_training:
            model.train()
        return out

    def _p_hgrads_for_edges(self, model: ToyTransformer,
                            edges: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """+dP/dh cotangent for the NTK pair pullback. Signed positive."""
        for p in model.parameters():
            p.grad = None
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, hiddens = model(edges, return_hidden=True)
            for h in hiddens:
                h.retain_grad()
            P = logits.float().gather(1, labels.unsqueeze(1)).sum()
        P.backward()
        grads = torch.stack([h.grad.detach().float() for h in hiddens], dim=0)
        for p in model.parameters():
            p.grad = None
        return grads

    def _run_ntk_pair_step(self, model: ToyTransformer) -> None:
        """Compute J_target J_source^T (-dL/dh_source) for four active/inactive
        source-target combinations, averaged over the fixed target subset.

        Stored per combo as (K, D, L, d_model) CPU tensors per probe step.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.func import functional_call, jvp, vjp

        was_training = model.training
        model.eval()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def hidden_fn(p, edges):
            _, hiddens = functional_call(
                model, (p, buffers), (edges,), {"return_hidden": True}
            )
            return torch.stack(hiddens, dim=0).float()

        for ell in self.target_layers:
            if ell not in self.ntk_pair_indices:
                continue
            K = self.Y_test[ell].shape[1]
            step_out = {
                "ta_sa": [], "ta_si": [], "ti_sa": [], "ti_si": []
            }
            step_out_p = {
                "ta_sa": [], "ta_si": [], "ti_sa": [], "ti_si": []
            }
            for k_idx in range(K):
                idx_a = self.ntk_pair_indices[ell]["active"][k_idx]
                idx_i = self.ntk_pair_indices[ell]["inactive"][k_idx]
                source_vecs = {}
                source_vecs_p = {}
                for s_name, source_idx in [("sa", idx_a), ("si", idx_i)]:
                    source_edges = self.edges[source_idx]
                    source_labels = self.nodes_sub[source_idx, -1]
                    cotangent = self._loss_hgrads_for_edges(
                        model, source_edges, source_labels)
                    with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
                        _, pullback = vjp(
                            lambda p: hidden_fn(p, source_edges), params)
                        source_vecs[s_name] = pullback(cotangent)[0]
                    del cotangent
                    cotangent_p = self._p_hgrads_for_edges(
                        model, source_edges, source_labels)
                    with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
                        _, pullback_p = vjp(
                            lambda p: hidden_fn(p, source_edges), params)
                        source_vecs_p[s_name] = pullback_p(cotangent_p)[0]
                    del cotangent_p

                for t_name, target_idx in [("ta", idx_a), ("ti", idx_i)]:
                    target_edges = self.edges[target_idx]
                    for s_name, tangent in source_vecs.items():
                        with sdpa_kernel(SDPBackend.MATH), torch.enable_grad(), \
                             torch.autocast("cuda", dtype=torch.bfloat16):
                            _, dh = jvp(
                                lambda p: hidden_fn(p, target_edges),
                                (params,), (tangent,)
                            )
                        combo = f"{t_name}_{s_name}"
                        step_out[combo].append(dh.detach().float().mean(dim=1).cpu())
                        del dh
                    for s_name, tangent in source_vecs_p.items():
                        with sdpa_kernel(SDPBackend.MATH), torch.enable_grad(), \
                             torch.autocast("cuda", dtype=torch.bfloat16):
                            _, dh_p = jvp(
                                lambda p: hidden_fn(p, target_edges),
                                (params,), (tangent,)
                            )
                        combo = f"{t_name}_{s_name}"
                        step_out_p[combo].append(dh_p.detach().float().mean(dim=1).cpu())
                        del dh_p
                    del target_edges
                del source_vecs, source_vecs_p

            for combo, values in step_out.items():
                if not hasattr(self, "ntk_pair_hist"):
                    self.ntk_pair_hist = {
                        key: {e: [] for e in self.target_layers}
                        for key in ("ta_sa", "ta_si", "ti_sa", "ti_si")
                    }
                self.ntk_pair_hist[combo][ell].append(
                    torch.stack(values, dim=0)  # (K, D, L, d)
                )
            for combo, values in step_out_p.items():
                self.ntk_pair_p_hist[combo][ell].append(
                    torch.stack(values, dim=0)  # (K, D, L, d)
                )
        if was_training:
            model.train()

    def _compute_final_direction_ntk_pair(
        self, model: ToyTransformer, ell: int, W_final: torch.Tensor
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute K_block(t) w_c^final for source/target active-inactive blocks.

        For each saved probe-time model state and class c:
            source tangent = J_source^T w_c^final
            target output  = J_target source_tangent

        Returns two nested combo dicts, each value shaped (T, D, L, K):
        cosine with w_c^final and raw dot with w_c^final.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.func import functional_call, jvp, vjp

        combos = ("ta_sa", "ta_si", "ti_sa", "ti_si")
        T = len(self.ntk_state_history)
        D, L, K, d_model = W_final.shape
        cos_out = {
            combo: torch.empty(T, D, L, K, dtype=torch.float32)
            for combo in combos
        }
        raw_out = {
            combo: torch.empty(T, D, L, K, dtype=torch.float32)
            for combo in combos
        }
        if T == 0 or ell not in self.ntk_pair_indices:
            return cos_out, raw_out

        was_training = model.training
        final_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        W_unit = torch.nn.functional.normalize(W_final, dim=-1, eps=1e-12)

        def hidden_fn(p, buffers, edges):
            _, hiddens = functional_call(
                model, (p, buffers), (edges,), {"return_hidden": True}
            )
            return torch.stack(hiddens, dim=0).float()

        model.eval()
        for t, state in enumerate(self.ntk_state_history):
            model.load_state_dict(
                {name: tensor.to(self.device) for name, tensor in state.items()}
            )
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            for k_idx in range(K):
                idx_a = self.ntk_pair_indices[ell]["active"][k_idx]
                idx_i = self.ntk_pair_indices[ell]["inactive"][k_idx]
                source_tangents = {}
                for s_name, source_idx in [("sa", idx_a), ("si", idx_i)]:
                    source_edges = self.edges[source_idx]
                    source_n = source_edges.shape[0]
                    cotangent = (
                        W_final[:, :, k_idx, :]
                        .unsqueeze(1)
                        .expand(D, source_n, L, d_model)
                        .contiguous()
                        / max(source_n, 1)
                    )
                    with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
                        _, pullback = vjp(
                            lambda p: hidden_fn(p, buffers, source_edges),
                            params,
                        )
                        source_tangents[s_name] = pullback(cotangent)[0]
                    del cotangent

                for t_name, target_idx in [("ta", idx_a), ("ti", idx_i)]:
                    target_edges = self.edges[target_idx]
                    for s_name, tangent in source_tangents.items():
                        with sdpa_kernel(SDPBackend.MATH), \
                             torch.autocast("cuda", dtype=torch.bfloat16):
                            _, dh = jvp(
                                lambda p: hidden_fn(p, buffers, target_edges),
                                (params,),
                                (tangent,),
                            )
                        dh_mean = dh.detach().float().mean(dim=1)  # (D, L, d)
                        combo = f"{t_name}_{s_name}"
                        raw = (dh_mean * W_final[:, :, k_idx, :]).sum(dim=-1)
                        dh_unit = torch.nn.functional.normalize(
                            dh_mean, dim=-1, eps=1e-12)
                        cos = (dh_unit * W_unit[:, :, k_idx, :]).sum(dim=-1)
                        raw_out[combo][t, :, :, k_idx] = raw.cpu()
                        cos_out[combo][t, :, :, k_idx] = cos.cpu()
                        del dh, dh_mean, raw, cos, dh_unit
                    del target_edges
                del source_tangents

        model.load_state_dict(
            {name: tensor.to(self.device) for name, tensor in final_state.items()}
        )
        if was_training:
            model.train()
        return cos_out, raw_out

    @torch.no_grad()
    def _multiclass_dom_top(self, H: torch.Tensor, ell: int
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-class DoM top-1/top-5 accuracy on features H (D, N, L, d).
        Returns (top1, top5), each (D, L). Same scheme as run()'s inline block."""
        Y_tr_mc = self.Y_train_mc[ell]
        n_pos_mc = self.n_pos_mc[ell].unsqueeze(1)
        n_neg_mc = self.n_neg_mc[ell].unsqueeze(1)
        y_te_mc = self.y_test_full[ell]
        C = Y_tr_mc.shape[1]
        D, _, L, _ = H.shape
        k5 = min(5, C)
        top1 = torch.empty(D, L, dtype=torch.float32)
        top5 = torch.empty(D, L, dtype=torch.float32)
        for d in range(D):
            for p in range(L):
                X_tr = H[d, self.train_idx, p, :]
                X_te = H[d, self.test_idx, p, :]
                sum_all = X_tr.sum(0, keepdim=True)
                sum_pos = Y_tr_mc.T @ X_tr
                sum_neg = sum_all - sum_pos
                W_mc = sum_pos / n_pos_mc - sum_neg / n_neg_mc
                scores_mc = X_te @ W_mc.T
                top1[d, p] = (scores_mc.argmax(dim=1) == y_te_mc).float().mean().cpu()
                top5[d, p] = (scores_mc.topk(k5, dim=1).indices
                              == y_te_mc.unsqueeze(1)).any(1).float().mean().cpu()
        return top1, top5

    @torch.no_grad()
    def _binary_dom_cube(self, H: torch.Tensor, ell: int) -> torch.Tensor:
        """DoM AUC on features H (D, N, L, d) for graph layer ell. Returns (D, L, K)."""
        Y_tr = self.Y_train[ell]
        Y_te = self.Y_test[ell]
        n_pos = self.n_pos_tr[ell].unsqueeze(1)
        n_neg = self.n_neg_tr[ell].unsqueeze(1)
        D, _, L, _ = H.shape
        K = Y_tr.shape[1]
        cube = torch.empty(D, L, K, dtype=torch.float32)
        for d in range(D):
            for p in range(L):
                X_tr = H[d, self.train_idx, p, :]
                X_te = H[d, self.test_idx, p, :]
                sum_all = X_tr.sum(0, keepdim=True)
                sum_pos = Y_tr.T @ X_tr
                sum_neg = sum_all - sum_pos
                w = sum_pos / n_pos - sum_neg / n_neg
                scores = X_te @ w.T
                cube[d, p] = _binary_auc_columns(scores, Y_te).cpu()
        return cube

    @torch.no_grad()
    def tick(self, model: ToyTransformer, step: int) -> None:
        """Called every training step. Caches h(t-1), h(t-10), and h(t-50) before a
        probe, finalizes actual update deltas at the probe step, then runs the
        probe."""
        if (self.pending_deriv_step >= 0
                and step == self.pending_deriv_step):
            was_training = model.training
            h_next = self._collect_hiddens(model)
            deriv = h_next - self.h_pending
            # Snapshot test-only slice on CPU for the dot-product analysis.
            self.deriv_test_history.append(
                deriv[:, self.test_idx, :, :].contiguous().cpu()
            )
            self.deriv_steps.append(step)
            for ell in self.target_layers:
                self.history["cubes_deriv"][ell].append(
                    self._binary_dom_cube(deriv, ell)
                )
                t1, t5 = self._multiclass_dom_top(deriv, ell)
                self.history["top1_deriv"][ell].append(t1)
                self.history["top5_deriv"][ell].append(t5)
            self.deriv_full_last = deriv.detach()
            del h_next, deriv
            self.h_pending = None
            self.pending_deriv_step = -1
            if was_training:
                model.train()
        if (self.pending_deriv10_step >= 0
                and step == self.pending_deriv10_step):
            was_training = model.training
            h_next = self._collect_hiddens(model)
            deriv10 = h_next - self.h_pending10
            self.deriv10_test_history.append(
                deriv10[:, self.test_idx, :, :].contiguous().cpu()
            )
            self.deriv10_steps.append(step)
            for ell in self.target_layers:
                t1, t5 = self._multiclass_dom_top(deriv10, ell)
                self.history["top1_deriv10"][ell].append(t1)
                self.history["top5_deriv10"][ell].append(t5)
            self.deriv10_full_last = deriv10.detach()
            del h_next, deriv10
            self.h_pending10 = None
            self.pending_deriv10_step = -1
            if was_training:
                model.train()
        if (self.pending_deriv50_step >= 0
                and step == self.pending_deriv50_step):
            was_training = model.training
            h_next = self._collect_hiddens(model)
            deriv50 = h_next - self.h_pending50
            self.deriv50_test_history.append(
                deriv50[:, self.test_idx, :, :].contiguous().cpu()
            )
            self.deriv50_steps.append(step)
            for ell in self.target_layers:
                t1, t5 = self._multiclass_dom_top(deriv50, ell)
                self.history["top1_deriv50"][ell].append(t1)
                self.history["top5_deriv50"][ell].append(t5)
            self.deriv50_full_last = deriv50.detach()
            del h_next, deriv50
            self.h_pending50 = None
            self.pending_deriv50_step = -1
            if was_training:
                model.train()
        if (self.cfg.every_steps > 0
                and step > 0
                and (step + 1) % self.cfg.every_steps == 0):
            was_training = model.training
            self.h_pending = self._collect_hiddens(model)
            self.pending_deriv_step = step + 1
            if was_training:
                model.train()
        if (self.cfg.every_steps > 0
                and step > 0
                and (step + 10) % self.cfg.every_steps == 0):
            was_training = model.training
            self.h_pending10 = self._collect_hiddens(model)
            self.pending_deriv10_step = step + 10
            if was_training:
                model.train()
        if (self.cfg.every_steps > 0
                and step > 0
                and (step + 50) % self.cfg.every_steps == 0):
            was_training = model.training
            self.h_pending50 = self._collect_hiddens(model)
            self.pending_deriv50_step = step + 50
            if was_training:
                model.train()
        if step % self.cfg.every_steps == 0:
            self.run(model, step)

    @torch.no_grad()
    def run(self, model: ToyTransformer, step: int) -> None:
        was_training = model.training
        hiddens = self._collect_hiddens(model)
        self.h_full_last = hiddens.detach()
        # Snapshot h(t) on the test subset for the end-of-training dot-product analysis.
        h_test_now = hiddens[:, self.test_idx, :, :].contiguous()
        self.h_test_history.append(h_test_now.cpu())
        self._h_test_now = h_test_now
        N_sub = self.edges.shape[0]
        loss_hgrads_full = self._collect_loss_hgrads(
            model, indices=torch.arange(N_sub, device=self.device))
        loss_hgrads_test = loss_hgrads_full[:, self.test_idx, :, :].contiguous()
        self.loss_grad_test_history.append(loss_hgrads_test.cpu())
        for ell in self.target_layers:
            t1_lg, t5_lg = self._multiclass_dom_top(loss_hgrads_full, ell)
            self.history["top1_loss_grad"][ell].append(t1_lg)
            self.history["top5_loss_grad"][ell].append(t5_lg)
        self.loss_grad_full_last = loss_hgrads_full.detach()
        del loss_hgrads_test
        self.ntk_state_history.append({
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        })
        self._run_ntk_pair_step(model)
        D, _, L, _ = hiddens.shape
        for ell in self.target_layers:
            Y_tr = self.Y_train[ell]                  # (Ntr, K)
            Y_te = self.Y_test[ell]                   # (Nte, K)
            n_pos = self.n_pos_tr[ell].unsqueeze(1)   # (K, 1)
            n_neg = self.n_neg_tr[ell].unsqueeze(1)   # (K, 1)
            K = Y_tr.shape[1]
            Y_tr_mc = self.Y_train_mc[ell]            # (Ntr, C)
            n_pos_mc = self.n_pos_mc[ell].unsqueeze(1)
            n_neg_mc = self.n_neg_mc[ell].unsqueeze(1)
            y_te_mc = self.y_test_full[ell]           # (Nte,)
            C = Y_tr_mc.shape[1]
            k5 = min(5, C)
            cube = torch.empty(D, L, K, dtype=torch.float32)
            top1 = torch.empty(D, L, dtype=torch.float32)
            top5 = torch.empty(D, L, dtype=torch.float32)
            for d in range(D):
                for p in range(L):
                    X_tr = hiddens[d, self.train_idx, p, :]
                    X_te = hiddens[d, self.test_idx, p, :]
                    sum_all = X_tr.sum(0, keepdim=True)
                    # Binary DoM AUC over K sampled nodes
                    sum_pos = Y_tr.T @ X_tr                 # (K, d)
                    sum_neg = sum_all - sum_pos
                    w = sum_pos / n_pos - sum_neg / n_neg   # (K, d)
                    scores = X_te @ w.T                     # (Nte, K)
                    cube[d, p] = _binary_auc_columns(scores, Y_te).cpu()
                    # Multi-class DoM over ALL reachable classes at layer ell
                    sum_pos_mc = Y_tr_mc.T @ X_tr           # (C, d)
                    sum_neg_mc = sum_all - sum_pos_mc
                    W_mc = sum_pos_mc / n_pos_mc - sum_neg_mc / n_neg_mc   # (C, d)
                    scores_mc = X_te @ W_mc.T               # (Nte, C)
                    top1[d, p] = (scores_mc.argmax(dim=1) == y_te_mc).float().mean().cpu()
                    top5[d, p] = (scores_mc.topk(k5, dim=1).indices
                                  == y_te_mc.unsqueeze(1)).any(1).float().mean().cpu()
            self.history["cubes"][ell].append(cube)
            self.history["top1"][ell].append(top1)
            self.history["top5"][ell].append(top5)

            if self.cfg.mlp_hidden > 0:
                mlp_top1, mlp_top5 = self._run_mlp_probes(hiddens, ell, D, L)
                self.history["mlp_top1"][ell].append(mlp_top1)
                self.history["mlp_top5"][ell].append(mlp_top5)

            if self.cfg.logreg_iters > 0:
                lr_top1, lr_top5 = self._run_logreg_probes(hiddens, ell, D, L)
                self.history["logreg_top1"][ell].append(lr_top1)
                self.history["logreg_top5"][ell].append(lr_top5)

            if self.cfg.steer_n_classes > 0:
                self._run_steering(model, hiddens, ell, top1, D, L)

        # Diff line: DoM on (h(t) - h(prev probe)). First probe: h_prev==h -> zeros.
        if self.h_prev_probe is None:
            self.h_prev_probe = hiddens
        diff = hiddens - self.h_prev_probe
        for ell in self.target_layers:
            self.history["cubes_diff"][ell].append(
                self._binary_dom_cube(diff, ell)
            )
            t1, t5 = self._multiclass_dom_top(diff, ell)
            self.history["top1_diff"][ell].append(t1)
            self.history["top5_diff"][ell].append(t5)
        self.diff_full_last = diff.detach()
        del diff

        # Update prev-probe cache for the next probe's diff.
        self.h_prev_probe = hiddens

        self._h_test_now = None

        self.history["step"].append(step)
        if was_training:
            model.train()

    def _run_logreg_probes(self, hiddens: torch.Tensor, ell: int,
                           D: int, L: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit multinomial logistic regressions at every (d, p) cell, batched."""
        cfg = self.cfg
        n_tr = self.train_idx.numel()
        cap = min(cfg.logreg_max_train, n_tr) if cfg.logreg_max_train > 0 else n_tr
        tr_sub = self.train_idx[:cap]
        te = self.test_idx
        d_model = hiddens.shape[-1]
        X_tr = hiddens[:, tr_sub, :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d_model)
        X_te = hiddens[:, te,      :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d_model)
        y_tr = self.y_train_full[ell][:cap]
        y_te = self.y_test_full[ell]
        C = self.reachable_classes[ell].numel()
        top1_flat, top5_flat = _fit_batched_logreg(
            X_tr, y_tr, X_te, y_te,
            n_iters=cfg.logreg_iters, lr=cfg.logreg_lr, n_classes=C,
        )
        return top1_flat.view(D, L), top5_flat.view(D, L)

    def _run_mlp_probes(self, hiddens: torch.Tensor, ell: int,
                        D: int, L: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit 1-hidden-layer MLP probes at every (d, p) cell for graph layer ell.
        Batched via bmm across all D*L cells. Returns top1, top5 tensors (D, L)."""
        cfg = self.cfg
        # Subsample train examples for MLP fit (cheaper).
        n_tr = self.train_idx.numel()
        cap = min(cfg.mlp_max_train, n_tr) if cfg.mlp_max_train > 0 else n_tr
        # Deterministic subsample: first `cap` rows of the probe train subset.
        tr_sub = self.train_idx[:cap]
        te = self.test_idx
        d_model = hiddens.shape[-1]
        # (D, N_sub, L, d) -> (D, L, N_sub, d) -> (D*L, N_sub, d)
        X_tr = hiddens[:, tr_sub, :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d_model)
        X_te = hiddens[:, te,      :, :].permute(0, 2, 1, 3).reshape(D * L, -1, d_model)
        y_tr = self.y_train_full[ell][:cap]
        y_te = self.y_test_full[ell]
        C = self.reachable_classes[ell].numel()
        top1_flat, top5_flat = _fit_batched_mlp(
            X_tr, y_tr, X_te, y_te,
            hidden=cfg.mlp_hidden, n_iters=cfg.mlp_iters, lr=cfg.mlp_lr,
            activation=cfg.mlp_activation, n_classes=C,
        )
        return top1_flat.view(D, L), top5_flat.view(D, L)

    @torch.no_grad()
    def _run_probe_grad_step(self, model: ToyTransformer) -> None:
        """Per (frac, ell, c): take one virtual SGD step on the fixed probe
        batch for c and record class-conditioned delta_h means on the test
        subset.

        Baseline h(t) on test is self._h_test_now (set by run()). At each c we:
          1) snapshot trainable params;
          2) grad the CE loss on the pre-sampled batch for c;
          3) apply p.data -= lr * p.grad in place;
          4) forward on the test-subset edges to get h_new_test;
          5) restore params, zero grads;
          6) accumulate delta = h_new_test - self._h_test_now, take active-c
             and inactive-c means over test samples.
        Params, grads, and optimizer state are untouched at return.
        """
        assert self._h_test_now is not None
        edges_test = self.edges[self.test_idx]
        h_base = self._h_test_now                                  # (D, Nte, L, d)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        state = {n: p.data.clone() for n, p in trainable}
        was_training = model.training
        lr = self.cfg.probe_grad_lr
        for frac in self.cfg.probe_grad_active_fracs:
            for ell in self.target_layers:
                Y_te = self.Y_test[ell]                            # (Nte, K)
                K = Y_te.shape[1]
                step_active = []
                step_inactive = []
                for k_idx in range(K):
                    batch_idx = self.probe_grad_batches[frac][ell][k_idx]
                    edges_batch = self.edges[batch_idx]
                    labels_batch = self.nodes_sub[batch_idx, -1]
                    for _, p in trainable:
                        p.grad = None
                    model.train()
                    with torch.enable_grad(), \
                         torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(edges_batch)
                        loss = torch.nn.functional.cross_entropy(
                            logits.float(), labels_batch)
                    loss.backward()
                    with torch.no_grad():
                        for _, p in trainable:
                            if p.grad is not None:
                                p.data.add_(p.grad, alpha=-lr)
                    model.eval()
                    with torch.no_grad():
                        h_new_test = self._forward_hiddens(model, edges_test)
                        delta = h_new_test - h_base                # (D, Nte, L, d)
                        Y_te_c = Y_te[:, k_idx].bool()
                        step_active.append(delta[:, Y_te_c, :, :].mean(dim=1).cpu())
                        step_inactive.append(delta[:, ~Y_te_c, :, :].mean(dim=1).cpu())
                    # Restore params
                    with torch.no_grad():
                        for n, p in trainable:
                            p.data.copy_(state[n])
                    for _, p in trainable:
                        p.grad = None
                self.probe_delta_active_hist[frac][ell].append(
                    torch.stack(step_active, dim=0)                # (K, D, L, d)
                )
                self.probe_delta_inactive_hist[frac][ell].append(
                    torch.stack(step_inactive, dim=0)
                )
        if was_training:
            model.train()

    def _run_steering(self, model: ToyTransformer, hiddens: torch.Tensor,
                      ell: int, top1: torch.Tensor, D: int, L: int) -> None:
        """One steering pass for graph layer `ell`, intervening at the current
        best DoM cell for that layer. Records mean-per-class success + baseline."""
        idx = top1.flatten().argmax().item()
        d_star, p_star = idx // L, idx % L
        targets = self.steer_targets[ell]                     # (Ks,)
        reach = self.steer_reachable[ell]                     # (Ks, C_out)
        Ks = targets.numel()

        # DoM directions at (d*, p*) computed on the probe train subset
        X_tr = hiddens[d_star, self.train_idx, p_star, :]     # (Ntr, d)
        y_tr = self.y_ell[ell][self.train_idx]                # (Ntr,) raw ids
        eq = (y_tr.unsqueeze(1) == targets.unsqueeze(0)).float()   # (Ntr, Ks)
        n_pos = eq.sum(0).clamp(min=1.0)
        n_neg = (X_tr.shape[0] - eq.sum(0)).clamp(min=1.0)
        sum_all = X_tr.sum(0, keepdim=True)
        sum_pos = eq.T @ X_tr                                  # (Ks, d)
        sum_neg = sum_all - sum_pos
        W = sum_pos / n_pos.unsqueeze(1) - sum_neg / n_neg.unsqueeze(1)   # (Ks, d)

        # Batched intervention: replicate test edges Ks times, add scaled w[k]
        # to the k-th block of rows.
        edges_te = self.edges[self.test_idx]                   # (Nte, seq_len)
        Nte = edges_te.shape[0]
        edges_rep = edges_te.repeat(Ks, 1)                     # (Ks*Nte, seq_len)
        vec_rep = (self.cfg.steer_alpha * W).repeat_interleave(Nte, dim=0)  # (Ks*Nte, d)
        logits = _forward_with_steer(model, edges_rep, vec_rep, d_star, p_star)
        preds = logits.argmax(dim=1).view(Ks, Nte)             # predicted node6 ids

        # Baseline: same forward without intervention (once, shared across Ks)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits_clean = model(edges_te)
        preds_clean = logits_clean.argmax(dim=1)               # (Nte,) node6 ids

        # Success = fraction whose predicted node6 is reachable from target class
        # in layer ell. Baseline uses the *unsteered* prediction against the same
        # reachable set (chance under the model's clean distribution).
        success = torch.empty(Ks, dtype=torch.float32)
        baseline = torch.empty(Ks, dtype=torch.float32)
        for k in range(Ks):
            success[k] = reach[k, preds[k]].float().mean().cpu()
            baseline[k] = reach[k, preds_clean].float().mean().cpu()

        self.history["steer"][ell].append(success)
        self.history["steer_baseline"][ell].append(baseline)
        self.history["steer_cell"][ell].append(
            torch.tensor([d_star, p_star], dtype=torch.long)
        )

    @torch.no_grad()
    def _aggregate_cosines(self, snapshots: list[torch.Tensor],
                           W_final: torch.Tensor, Y_te: torch.Tensor,
                           n_pos_te: torch.Tensor, n_neg_te: torch.Tensor,
                           with_std: bool = False,
                           ):
        """For each (D, Nte, L, d) snapshot X, score cosine(W_final[d,l,c,:],
        X[d,n,l,:]) per test sample and aggregate mean over active (y=c) and
        inactive (y!=c) samples per class. Returns (active, inactive), each
        (T, D, L, K). If with_std, also returns (active_std, inactive_std)
        computed as within-class sample std at each t."""
        T = len(snapshots)
        D, L, K, _ = W_final.shape
        active = torch.empty(T, D, L, K, dtype=torch.float32)
        inactive = torch.empty(T, D, L, K, dtype=torch.float32)
        if with_std:
            active_std = torch.empty(T, D, L, K, dtype=torch.float32)
            inactive_std = torch.empty(T, D, L, K, dtype=torch.float32)
        Y_te_ = Y_te.unsqueeze(0).unsqueeze(0)                  # (1, 1, Nte, K)
        W_unit = torch.nn.functional.normalize(W_final, dim=-1, eps=1e-12)
        for t, snap in enumerate(snapshots):
            X = snap.to(self.device)
            X_unit = torch.nn.functional.normalize(X, dim=-1, eps=1e-12)
            cos = torch.einsum("dlkm,dnlm->dlnk", W_unit, X_unit)  # (D, L, Nte, K)
            sum_pos = (cos * Y_te_).sum(dim=2)                    # (D, L, K)
            sum_all = cos.sum(dim=2)
            sum_neg = sum_all - sum_pos
            mean_pos = sum_pos / n_pos_te
            mean_neg = sum_neg / n_neg_te
            active[t] = mean_pos.cpu()
            inactive[t] = mean_neg.cpu()
            if with_std:
                cos_sq = cos * cos
                sum_sq_pos = (cos_sq * Y_te_).sum(dim=2)
                sum_sq_all = cos_sq.sum(dim=2)
                sum_sq_neg = sum_sq_all - sum_sq_pos
                var_pos = (sum_sq_pos / n_pos_te - mean_pos * mean_pos).clamp(min=0.0)
                var_neg = (sum_sq_neg / n_neg_te - mean_neg * mean_neg).clamp(min=0.0)
                active_std[t] = var_pos.sqrt().cpu()
                inactive_std[t] = var_neg.sqrt().cpu()
                del cos_sq
            del X, X_unit, cos
        if with_std:
            return active, inactive, active_std, inactive_std
        return active, inactive

    @torch.no_grad()
    def _aggregate_raw_dots(self, snapshots: list[torch.Tensor],
                            W_final: torch.Tensor, Y_te: torch.Tensor,
                            n_pos_te: torch.Tensor, n_neg_te: torch.Tensor,
                            with_std: bool = False,
                            ):
        """For each (D, Nte, L, d) snapshot X, score <W_final[d,l,c,:],
        X[d,n,l,:]> and aggregate mean over active/inactive samples per class.
        Returns (active, inactive), each (T, D, L, K). If with_std, also
        returns (active_std, inactive_std)."""
        T = len(snapshots)
        D, L, K, _ = W_final.shape
        active = torch.empty(T, D, L, K, dtype=torch.float32)
        inactive = torch.empty(T, D, L, K, dtype=torch.float32)
        if with_std:
            active_std = torch.empty(T, D, L, K, dtype=torch.float32)
            inactive_std = torch.empty(T, D, L, K, dtype=torch.float32)
        Y_te_ = Y_te.unsqueeze(0).unsqueeze(0)                  # (1, 1, Nte, K)
        for t, snap in enumerate(snapshots):
            X = snap.to(self.device)
            dots = torch.einsum("dlkm,dnlm->dlnk", W_final, X)  # (D, L, Nte, K)
            sum_pos = (dots * Y_te_).sum(dim=2)                 # (D, L, K)
            sum_all = dots.sum(dim=2)
            sum_neg = sum_all - sum_pos
            mean_pos = sum_pos / n_pos_te
            mean_neg = sum_neg / n_neg_te
            active[t] = mean_pos.cpu()
            inactive[t] = mean_neg.cpu()
            if with_std:
                dots_sq = dots * dots
                sum_sq_pos = (dots_sq * Y_te_).sum(dim=2)
                sum_sq_all = dots_sq.sum(dim=2)
                sum_sq_neg = sum_sq_all - sum_sq_pos
                var_pos = (sum_sq_pos / n_pos_te - mean_pos * mean_pos).clamp(min=0.0)
                var_neg = (sum_sq_neg / n_neg_te - mean_neg * mean_neg).clamp(min=0.0)
                active_std[t] = var_pos.sqrt().cpu()
                inactive_std[t] = var_neg.sqrt().cpu()
                del dots_sq
            del X, dots
        if with_std:
            return active, inactive, active_std, inactive_std
        return active, inactive

    @torch.no_grad()
    def _aggregate_norms(self, snapshots: list[torch.Tensor],
                         Y_te: torch.Tensor,
                         n_pos_te: torch.Tensor, n_neg_te: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """For each (D, Nte, L, d) snapshot X, aggregate ||X[d,n,l,:]|| by
        active/inactive class membership. Returns (active, inactive), each
        (T, D, L, K)."""
        T = len(snapshots)
        K = Y_te.shape[1]
        if T == 0:
            return torch.empty(0), torch.empty(0)
        D, _, L, _ = snapshots[0].shape
        active = torch.empty(T, D, L, K, dtype=torch.float32)
        inactive = torch.empty(T, D, L, K, dtype=torch.float32)
        Y_te_ = Y_te.unsqueeze(0).unsqueeze(0)                  # (1, 1, Nte, K)
        for t, snap in enumerate(snapshots):
            X = snap.to(self.device)
            norms = X.norm(dim=-1).permute(0, 2, 1).unsqueeze(-1)  # (D, L, Nte, 1)
            sum_pos = (norms * Y_te_).sum(dim=2)                  # (D, L, K)
            sum_all = norms.sum(dim=2).expand(-1, -1, K)
            sum_neg = sum_all - sum_pos
            active[t] = (sum_pos / n_pos_te).cpu()
            inactive[t] = (sum_neg / n_neg_te).cpu()
            del X, norms
        return active, inactive

    def _final_directions_at_cell(self, feature_full: torch.Tensor, ell: int,
                                  d_star: int, p_star: int):
        """Fit DoM-MC direction and multinomial-logreg direction at (d*, p*)
        from the given (D, N_sub, L, d_model) feature tensor. Returns both as
        (C, d) tensors on device."""
        X_tr = feature_full[d_star, self.train_idx, p_star, :].detach()
        Y_tr_mc = self.Y_train_mc[ell]                                # (Ntr, C)
        n_pos_mc = self.n_pos_mc[ell].unsqueeze(1)                    # (C, 1)
        n_neg_mc = self.n_neg_mc[ell].unsqueeze(1)
        with torch.no_grad():
            sum_all = X_tr.sum(0, keepdim=True)
            sum_pos_mc = Y_tr_mc.T @ X_tr
            sum_neg_mc = sum_all - sum_pos_mc
            W_dom_mc = sum_pos_mc / n_pos_mc - sum_neg_mc / n_neg_mc  # (C, d)
        y_tr = self.y_train_full[ell]
        C = self.reachable_classes[ell].numel()
        W_lr = self._fit_multinomial_logreg_at_cell(X_tr, y_tr, C)   # (C, d)
        return W_dom_mc, W_lr

    @torch.no_grad()
    def _final_applied_curve(self, snapshots: list, W: torch.Tensor,
                             ell: int, d_star: int, p_star: int) -> torch.Tensor:
        """Score X_te(t) @ W.T at cell (d*, p*) on each stored snapshot, return
        (T,) top-1 curve using the multiclass label vector."""
        y_te = self.y_test_full[ell]
        T = len(snapshots)
        curve = torch.empty(T, dtype=torch.float32)
        for t, snap in enumerate(snapshots):
            X_te = snap.to(self.device)[d_star, :, p_star, :]
            scores = X_te @ W.T
            curve[t] = (scores.argmax(dim=1) == y_te).float().mean().cpu()
        return curve

    def _fit_multinomial_logreg_at_cell(self, X_tr: torch.Tensor,
                                        y_tr: torch.Tensor,
                                        C: int) -> torch.Tensor:
        """Multinomial softmax logistic regression over C classes.
        X_tr: (Ntr, d). y_tr: (Ntr,) with labels in [0, C). Returns W: (C, d).

        Mirrors the multiclass DoM target space (all reachable classes).
        Extract per-probe-class directions afterwards.
        """
        Ntr, d = X_tr.shape
        device = X_tr.device
        with torch.enable_grad():
            W = torch.zeros(C, d, device=device, requires_grad=True)
            b = torch.zeros(C, device=device, requires_grad=True)
            optim = torch.optim.Adam([W, b], lr=self.cfg.lr_direction_lr)
            for _ in range(self.cfg.lr_direction_iters):
                logits = X_tr @ W.T + b                      # (Ntr, C)
                loss = torch.nn.functional.cross_entropy(
                    logits, y_tr, reduction="mean")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
        return W.detach()

    @torch.no_grad()
    def _aggregate_cosines_at_cell(self, snapshots: list,
                                   w: torch.Tensor,
                                   d_star: int, p_star: int,
                                   Y_te: torch.Tensor,
                                   n_pos_te: torch.Tensor,
                                   n_neg_te: torch.Tensor,
                                   with_std: bool = False):
        """snapshots: list of (D, Nte, L, d_model). w: (K, d_model). Returns
        active, inactive each (T, K)."""
        T = len(snapshots)
        K = w.shape[0]
        active = torch.empty(T, K, dtype=torch.float32)
        inactive = torch.empty(T, K, dtype=torch.float32)
        if with_std:
            active_std = torch.empty(T, K, dtype=torch.float32)
            inactive_std = torch.empty(T, K, dtype=torch.float32)
        w_unit = torch.nn.functional.normalize(w, dim=-1, eps=1e-12)
        for t, snap in enumerate(snapshots):
            X = snap.to(self.device)[d_star, :, p_star, :]         # (Nte, d)
            X_unit = torch.nn.functional.normalize(X, dim=-1, eps=1e-12)
            cos = X_unit @ w_unit.T                                # (Nte, K)
            sum_pos = (cos * Y_te).sum(dim=0)                      # (K,)
            sum_all = cos.sum(dim=0)
            sum_neg = sum_all - sum_pos
            mean_pos = sum_pos / n_pos_te
            mean_neg = sum_neg / n_neg_te
            active[t] = mean_pos.cpu()
            inactive[t] = mean_neg.cpu()
            if with_std:
                cos_sq = cos * cos
                sum_sq_pos = (cos_sq * Y_te).sum(dim=0)
                sum_sq_neg = cos_sq.sum(dim=0) - sum_sq_pos
                var_pos = (sum_sq_pos / n_pos_te - mean_pos * mean_pos).clamp(min=0.0)
                var_neg = (sum_sq_neg / n_neg_te - mean_neg * mean_neg).clamp(min=0.0)
                active_std[t] = var_pos.sqrt().cpu()
                inactive_std[t] = var_neg.sqrt().cpu()
        if with_std:
            return active, inactive, active_std, inactive_std
        return active, inactive

    @torch.no_grad()
    def _aggregate_raw_dots_at_cell(self, snapshots: list,
                                    w: torch.Tensor,
                                    d_star: int, p_star: int,
                                    Y_te: torch.Tensor,
                                    n_pos_te: torch.Tensor,
                                    n_neg_te: torch.Tensor,
                                    with_std: bool = False):
        """snapshots: list of (D, Nte, L, d_model). w: (K, d_model). Returns
        active, inactive each (T, K)."""
        T = len(snapshots)
        K = w.shape[0]
        active = torch.empty(T, K, dtype=torch.float32)
        inactive = torch.empty(T, K, dtype=torch.float32)
        if with_std:
            active_std = torch.empty(T, K, dtype=torch.float32)
            inactive_std = torch.empty(T, K, dtype=torch.float32)
        for t, snap in enumerate(snapshots):
            X = snap.to(self.device)[d_star, :, p_star, :]         # (Nte, d)
            dots = X @ w.T                                         # (Nte, K)
            sum_pos = (dots * Y_te).sum(dim=0)
            sum_all = dots.sum(dim=0)
            sum_neg = sum_all - sum_pos
            mean_pos = sum_pos / n_pos_te
            mean_neg = sum_neg / n_neg_te
            active[t] = mean_pos.cpu()
            inactive[t] = mean_neg.cpu()
            if with_std:
                dots_sq = dots * dots
                sum_sq_pos = (dots_sq * Y_te).sum(dim=0)
                sum_sq_neg = dots_sq.sum(dim=0) - sum_sq_pos
                var_pos = (sum_sq_pos / n_pos_te - mean_pos * mean_pos).clamp(min=0.0)
                var_neg = (sum_sq_neg / n_neg_te - mean_neg * mean_neg).clamp(min=0.0)
                active_std[t] = var_pos.sqrt().cpu()
                inactive_std[t] = var_neg.sqrt().cpu()
        if with_std:
            return active, inactive, active_std, inactive_std
        return active, inactive

    @torch.no_grad()
    def finalize_dot_products(self, model: ToyTransformer) -> None:
        """Compute cos(w_c^final, X(t)) per test sample and aggregate by class
        membership, for X in three flavors:
            rep     : X = h(t)
            diff    : X = h(t) - h(prev probe step)     (first entry: zeros)
            deriv   : X = h(t) - h(t - 1 training step)

        w_c^final is the DoM direction at every (d, p) computed from h_final on
        the train subset. Populates dot_{rep,diff,deriv}_{active,inactive}[ell]
        with cosine values as (T, D, L, K) tensors, plus final_top1_rep[ell]
        as (D, L).
        """
        if not self.h_test_history:
            return
        was_training = model.training
        h_final = self._collect_hiddens(model)                  # (D, N, L, d_model)
        D, _, L, d_model = h_final.shape

        # Precompute diff snapshots: h_test[t] - h_test[t-1], with t=0 -> zeros.
        diff_snapshots: list[torch.Tensor] = []
        for t, h_t in enumerate(self.h_test_history):
            if t == 0:
                diff_snapshots.append(torch.zeros_like(h_t))
            else:
                diff_snapshots.append(h_t - self.h_test_history[t - 1])

        for ell in self.target_layers:
            Y_tr = self.Y_train[ell]                            # (Ntr, K)
            Y_te = self.Y_test[ell]                             # (Nte, K)
            n_pos_tr = self.n_pos_tr[ell]                       # (K,)
            n_neg_tr = self.n_neg_tr[ell]
            K = Y_tr.shape[1]

            # Final DoM direction at every (d, p): W_final (D, L, K, d_model).
            W_final = torch.empty(D, L, K, d_model, device=self.device)
            for d in range(D):
                for p in range(L):
                    X_tr = h_final[d, self.train_idx, p, :]     # (Ntr, d_model)
                    sum_all = X_tr.sum(0, keepdim=True)
                    sum_pos = Y_tr.T @ X_tr
                    sum_neg = sum_all - sum_pos
                    W_final[d, p] = (sum_pos / n_pos_tr.unsqueeze(1)
                                     - sum_neg / n_neg_tr.unsqueeze(1))

            t1_final, _ = self._multiclass_dom_top(h_final, ell)
            self.final_top1_rep[ell] = t1_final                 # (D, L)

            n_pos_te = Y_te.sum(0).clamp(min=1.0)
            n_neg_te = (Y_te.shape[0] - Y_te.sum(0)).clamp(min=1.0)

            rep_a, rep_i = self._aggregate_cosines(
                self.h_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            diff_a, diff_i = self._aggregate_cosines(
                diff_snapshots, W_final, Y_te, n_pos_te, n_neg_te)
            deriv_a, deriv_i = self._aggregate_cosines(
                self.deriv_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            deriv10_a, deriv10_i = self._aggregate_cosines(
                self.deriv10_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            deriv50_a, deriv50_i = self._aggregate_cosines(
                self.deriv50_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            loss_grad_a, loss_grad_i, loss_grad_a_std, loss_grad_i_std = (
                self._aggregate_cosines(
                    self.loss_grad_test_history, W_final, Y_te,
                    n_pos_te, n_neg_te, with_std=True)
            )
            jvp_update_a, jvp_update_i = self._aggregate_cosines(
                self.jvp_update_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            raw_rep_a, raw_rep_i = self._aggregate_raw_dots(
                self.h_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            raw_diff_a, raw_diff_i = self._aggregate_raw_dots(
                diff_snapshots, W_final, Y_te, n_pos_te, n_neg_te)
            raw_deriv_a, raw_deriv_i = self._aggregate_raw_dots(
                self.deriv_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            raw_deriv10_a, raw_deriv10_i = self._aggregate_raw_dots(
                self.deriv10_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            raw_deriv50_a, raw_deriv50_i = self._aggregate_raw_dots(
                self.deriv50_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            (raw_loss_grad_a, raw_loss_grad_i,
             raw_loss_grad_a_std, raw_loss_grad_i_std) = (
                self._aggregate_raw_dots(
                    self.loss_grad_test_history, W_final, Y_te,
                    n_pos_te, n_neg_te, with_std=True)
            )
            raw_jvp_update_a, raw_jvp_update_i = self._aggregate_raw_dots(
                self.jvp_update_test_history, W_final, Y_te, n_pos_te, n_neg_te)
            norm_diff_a, norm_diff_i = self._aggregate_norms(
                diff_snapshots, Y_te, n_pos_te, n_neg_te)
            norm_deriv_a, norm_deriv_i = self._aggregate_norms(
                self.deriv_test_history, Y_te, n_pos_te, n_neg_te)
            norm_deriv10_a, norm_deriv10_i = self._aggregate_norms(
                self.deriv10_test_history, Y_te, n_pos_te, n_neg_te)
            norm_deriv50_a, norm_deriv50_i = self._aggregate_norms(
                self.deriv50_test_history, Y_te, n_pos_te, n_neg_te)
            self.dot_rep_active[ell], self.dot_rep_inactive[ell] = rep_a, rep_i
            self.dot_diff_active[ell], self.dot_diff_inactive[ell] = diff_a, diff_i
            self.dot_deriv_active[ell], self.dot_deriv_inactive[ell] = deriv_a, deriv_i
            self.dot_deriv10_active[ell], self.dot_deriv10_inactive[ell] = (
                deriv10_a, deriv10_i
            )
            self.dot_deriv50_active[ell], self.dot_deriv50_inactive[ell] = (
                deriv50_a, deriv50_i
            )
            self.dot_loss_grad_active[ell], self.dot_loss_grad_inactive[ell] = (
                loss_grad_a, loss_grad_i
            )
            self.dot_loss_grad_active_std[ell] = loss_grad_a_std
            self.dot_loss_grad_inactive_std[ell] = loss_grad_i_std
            self.dot_jvp_update_active[ell], self.dot_jvp_update_inactive[ell] = (
                jvp_update_a, jvp_update_i
            )
            self.raw_dot_rep_active[ell], self.raw_dot_rep_inactive[ell] = (
                raw_rep_a, raw_rep_i
            )
            self.raw_dot_diff_active[ell], self.raw_dot_diff_inactive[ell] = (
                raw_diff_a, raw_diff_i
            )
            self.raw_dot_deriv_active[ell], self.raw_dot_deriv_inactive[ell] = (
                raw_deriv_a, raw_deriv_i
            )
            self.raw_dot_deriv10_active[ell], self.raw_dot_deriv10_inactive[ell] = (
                raw_deriv10_a, raw_deriv10_i
            )
            self.raw_dot_deriv50_active[ell], self.raw_dot_deriv50_inactive[ell] = (
                raw_deriv50_a, raw_deriv50_i
            )
            self.raw_dot_loss_grad_active[ell], self.raw_dot_loss_grad_inactive[ell] = (
                raw_loss_grad_a, raw_loss_grad_i
            )
            self.raw_dot_loss_grad_active_std[ell] = raw_loss_grad_a_std
            self.raw_dot_loss_grad_inactive_std[ell] = raw_loss_grad_i_std
            self.raw_dot_jvp_update_active[ell], self.raw_dot_jvp_update_inactive[ell] = (
                raw_jvp_update_a, raw_jvp_update_i
            )
            W_unit = torch.nn.functional.normalize(W_final, dim=-1, eps=1e-12)
            for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si"):
                if self.ntk_pair_hist[combo][ell]:
                    ntk = torch.stack(self.ntk_pair_hist[combo][ell], dim=0
                                      ).to(self.device)          # (T, K, D, L, d)
                    ntk_unit = torch.nn.functional.normalize(ntk, dim=-1, eps=1e-12)
                    self.dot_ntk_grad_pair[combo][ell] = torch.einsum(
                        "dlkm,tkdlm->tdlk", W_unit, ntk_unit).cpu()
                    self.raw_dot_ntk_grad_pair[combo][ell] = torch.einsum(
                        "dlkm,tkdlm->tdlk", W_final, ntk).cpu()
                    del ntk, ntk_unit
            if ell in self.cfg.ntk_pair_layers:
                ntk_cos, ntk_raw = self._compute_final_direction_ntk_pair(
                    model, ell, W_final
                )
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si"):
                    self.dot_ntk_pair[combo][ell] = ntk_cos[combo]
                    self.raw_dot_ntk_pair[combo][ell] = ntk_raw[combo]

            best_idx = t1_final.reshape(-1).argmax().item()
            d_star, p_star = best_idx // L, best_idx % L
            self.lr_cell[ell] = torch.tensor([d_star, p_star], dtype=torch.long)
            X_tr_cell = h_final[d_star, self.train_idx, p_star, :].detach()
            reachable = self.reachable_classes[ell]
            C_ell = reachable.numel()
            y_tr_mc = self.y_train_full[ell]
            W_full = self._fit_multinomial_logreg_at_cell(
                X_tr_cell, y_tr_mc, C_ell)                        # (C, d)
            remap = torch.full((int(reachable.max().item()) + 1,), -1,
                               dtype=torch.long, device=self.device)
            remap[reachable] = torch.arange(C_ell, device=self.device)
            chosen_C = remap[self.chosen[ell]]                    # (K,) in [0, C)
            w_lr = W_full[chosen_C]                               # (K, d)
            self.W_lr_final[ell] = w_lr.cpu()

            (self.dot_lr_rep_active[ell],
             self.dot_lr_rep_inactive[ell]) = self._aggregate_cosines_at_cell(
                self.h_test_history, w_lr, d_star, p_star, Y_te, n_pos_te, n_neg_te)
            (self.dot_lr_diff_active[ell],
             self.dot_lr_diff_inactive[ell]) = self._aggregate_cosines_at_cell(
                diff_snapshots, w_lr, d_star, p_star, Y_te, n_pos_te, n_neg_te)
            (self.dot_lr_deriv_active[ell],
             self.dot_lr_deriv_inactive[ell]) = self._aggregate_cosines_at_cell(
                self.deriv_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.dot_lr_deriv10_active[ell],
             self.dot_lr_deriv10_inactive[ell]) = self._aggregate_cosines_at_cell(
                self.deriv10_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.dot_lr_deriv50_active[ell],
             self.dot_lr_deriv50_inactive[ell]) = self._aggregate_cosines_at_cell(
                self.deriv50_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.dot_lr_loss_grad_active[ell],
             self.dot_lr_loss_grad_inactive[ell],
             self.dot_lr_loss_grad_active_std[ell],
             self.dot_lr_loss_grad_inactive_std[ell]
             ) = self._aggregate_cosines_at_cell(
                self.loss_grad_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te, with_std=True)

            (self.raw_dot_lr_rep_active[ell],
             self.raw_dot_lr_rep_inactive[ell]) = self._aggregate_raw_dots_at_cell(
                self.h_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.raw_dot_lr_diff_active[ell],
             self.raw_dot_lr_diff_inactive[ell]) = self._aggregate_raw_dots_at_cell(
                diff_snapshots, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.raw_dot_lr_deriv_active[ell],
             self.raw_dot_lr_deriv_inactive[ell]) = self._aggregate_raw_dots_at_cell(
                self.deriv_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.raw_dot_lr_deriv10_active[ell],
             self.raw_dot_lr_deriv10_inactive[ell]) = self._aggregate_raw_dots_at_cell(
                self.deriv10_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.raw_dot_lr_deriv50_active[ell],
             self.raw_dot_lr_deriv50_inactive[ell]) = self._aggregate_raw_dots_at_cell(
                self.deriv50_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te)
            (self.raw_dot_lr_loss_grad_active[ell],
             self.raw_dot_lr_loss_grad_inactive[ell],
             self.raw_dot_lr_loss_grad_active_std[ell],
             self.raw_dot_lr_loss_grad_inactive_std[ell]
             ) = self._aggregate_raw_dots_at_cell(
                self.loss_grad_test_history, w_lr, d_star, p_star, Y_te,
                n_pos_te, n_neg_te, with_std=True)

            w_lr_unit = torch.nn.functional.normalize(w_lr, dim=-1, eps=1e-12)
            for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si"):
                if self.ntk_pair_hist[combo][ell]:
                    ntk = torch.stack(self.ntk_pair_hist[combo][ell], dim=0
                                      ).to(self.device)             # (T, K, D, L, d)
                    ntk_cell = ntk[:, :, d_star, p_star, :]          # (T, K, d)
                    ntk_unit_cell = torch.nn.functional.normalize(
                        ntk_cell, dim=-1, eps=1e-12)
                    self.dot_lr_ntk_grad_pair[combo][ell] = (
                        (w_lr_unit.unsqueeze(0) * ntk_unit_cell).sum(-1).cpu()
                    )
                    self.raw_dot_lr_ntk_grad_pair[combo][ell] = (
                        (w_lr.unsqueeze(0) * ntk_cell).sum(-1).cpu()
                    )
                    del ntk, ntk_cell, ntk_unit_cell

            if ell in self.cfg.ntk_pair_layers:
                w_lr_full = torch.zeros_like(W_final)
                w_lr_full[d_star, p_star] = w_lr
                ntk_cos_lr, ntk_raw_lr = self._compute_final_direction_ntk_pair(
                    model, ell, w_lr_full
                )
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si"):
                    self.dot_lr_ntk_pair[combo][ell] = (
                        ntk_cos_lr[combo][:, d_star, p_star, :]
                    )
                    self.raw_dot_lr_ntk_pair[combo][ell] = (
                        ntk_raw_lr[combo][:, d_star, p_star, :]
                    )
                del ntk_cos_lr, ntk_raw_lr, w_lr_full

            feature_last = {
                "rep": (self.h_full_last, self.h_test_history),
                "diff": (self.diff_full_last, diff_snapshots),
                "deriv": (self.deriv_full_last, self.deriv_test_history),
                "deriv10": (self.deriv10_full_last, self.deriv10_test_history),
                "deriv50": (self.deriv50_full_last, self.deriv50_test_history),
                "loss_grad": (self.loss_grad_full_last, self.loss_grad_test_history),
            }
            for name, (feat_full, test_snapshots) in feature_last.items():
                if feat_full is None or len(test_snapshots) == 0:
                    continue
                W_dom_mc, W_lr = self._final_directions_at_cell(
                    feat_full, ell, d_star, p_star)
                self.final_dom_top1_curve[name][ell] = self._final_applied_curve(
                    test_snapshots, W_dom_mc, ell, d_star, p_star)
                self.final_lr_top1_curve[name][ell] = self._final_applied_curve(
                    test_snapshots, W_lr, ell, d_star, p_star)

            self.norm_diff_active[ell], self.norm_diff_inactive[ell] = (
                norm_diff_a, norm_diff_i
            )
            self.norm_deriv_active[ell], self.norm_deriv_inactive[ell] = (
                norm_deriv_a, norm_deriv_i
            )
            self.norm_deriv10_active[ell], self.norm_deriv10_inactive[ell] = (
                norm_deriv10_a, norm_deriv10_i
            )
            self.norm_deriv50_active[ell], self.norm_deriv50_inactive[ell] = (
                norm_deriv50_a, norm_deriv50_i
            )

            # Virtual-grad probe deltas: for each fraction and t, we already
            # have per-class active/inactive means as (K, D, L, d) tensors, one
            # per probe step. Contract against W_final to get (T, D, L, K).
            for frac in self.cfg.probe_grad_active_fracs:
                if self.probe_delta_active_hist[frac][ell]:
                    pd_a = torch.stack(self.probe_delta_active_hist[frac][ell], dim=0
                                       ).to(self.device)          # (T, K, D, L, d)
                    pd_i = torch.stack(self.probe_delta_inactive_hist[frac][ell], dim=0
                                       ).to(self.device)
                    pd_a_unit = torch.nn.functional.normalize(pd_a, dim=-1, eps=1e-12)
                    pd_i_unit = torch.nn.functional.normalize(pd_i, dim=-1, eps=1e-12)
                    self.dot_probe_active[frac][ell] = torch.einsum(
                        "dlkm,tkdlm->tdlk", W_unit, pd_a_unit).cpu()
                    self.dot_probe_inactive[frac][ell] = torch.einsum(
                        "dlkm,tkdlm->tdlk", W_unit, pd_i_unit).cpu()

        del h_final, diff_snapshots
        if was_training:
            model.train()

    def save(self, path: Path) -> None:
        payload = {
            "step": torch.tensor(self.history["step"], dtype=torch.long),
            "deriv_step": torch.tensor(self.deriv_steps, dtype=torch.long),
            "deriv10_step": torch.tensor(self.deriv10_steps, dtype=torch.long),
            "deriv50_step": torch.tensor(self.deriv50_steps, dtype=torch.long),
            "jvp_update_step": torch.tensor(self.jvp_update_steps, dtype=torch.long),
            "cubes": {ell: torch.stack(v, dim=0)     # (T, D, L, K)
                      for ell, v in self.history["cubes"].items()},
            "cubes_diff": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                           for ell, v in self.history["cubes_diff"].items()},
            # cubes_deriv has no entry for the initial step because h(t)-h(t-1)
            # is undefined before the first training update.
            "cubes_deriv": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                            for ell, v in self.history["cubes_deriv"].items()},
            "top1": {ell: torch.stack(v, dim=0)      # (T, D, L)
                     for ell, v in self.history["top1"].items()},
            "top5": {ell: torch.stack(v, dim=0)
                     for ell, v in self.history["top5"].items()},
            "top1_diff": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                          for ell, v in self.history["top1_diff"].items()},
            "top5_diff": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                          for ell, v in self.history["top5_diff"].items()},
            "top1_deriv": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                           for ell, v in self.history["top1_deriv"].items()},
            "top5_deriv": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                           for ell, v in self.history["top5_deriv"].items()},
            "top1_deriv10": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                             for ell, v in self.history["top1_deriv10"].items()},
            "top5_deriv10": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                             for ell, v in self.history["top5_deriv10"].items()},
            "top1_deriv50": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                             for ell, v in self.history["top1_deriv50"].items()},
            "top5_deriv50": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                             for ell, v in self.history["top5_deriv50"].items()},
            "top1_loss_grad": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                               for ell, v in self.history["top1_loss_grad"].items()},
            "top5_loss_grad": {ell: torch.stack(v, dim=0) if v else torch.empty(0)
                               for ell, v in self.history["top5_loss_grad"].items()},
            "chosen": {ell: v.cpu() for ell, v in self.chosen.items()},
            "reachable_classes": {ell: v.cpu() for ell, v in self.reachable_classes.items()},
            "target_layers": self.target_layers,
            "cfg": {"n_nodes": self.cfg.n_nodes,
                    "every_steps": self.cfg.every_steps,
                    "sample_seed": self.cfg.sample_seed,
                    "steer_n_classes": self.cfg.steer_n_classes,
                    "steer_alpha": self.cfg.steer_alpha},
        }
        if self.dot_rep_active:
            # cos(w_c^final, X(t)): mean over test samples split by whether
            # y_layer[x] == c. Shape (T, D, L, K) per target layer.
            payload["dot_rep_active"] = dict(self.dot_rep_active)
            payload["dot_rep_inactive"] = dict(self.dot_rep_inactive)
            payload["dot_diff_active"] = dict(self.dot_diff_active)
            payload["dot_diff_inactive"] = dict(self.dot_diff_inactive)
            payload["dot_deriv_active"] = dict(self.dot_deriv_active)
            payload["dot_deriv_inactive"] = dict(self.dot_deriv_inactive)
            payload["dot_deriv10_active"] = dict(self.dot_deriv10_active)
            payload["dot_deriv10_inactive"] = dict(self.dot_deriv10_inactive)
            payload["dot_deriv50_active"] = dict(self.dot_deriv50_active)
            payload["dot_deriv50_inactive"] = dict(self.dot_deriv50_inactive)
            payload["dot_loss_grad_active"] = dict(self.dot_loss_grad_active)
            payload["dot_loss_grad_inactive"] = dict(self.dot_loss_grad_inactive)
            payload["dot_loss_grad_active_std"] = dict(self.dot_loss_grad_active_std)
            payload["dot_loss_grad_inactive_std"] = dict(self.dot_loss_grad_inactive_std)
            payload["dot_jvp_update_active"] = dict(self.dot_jvp_update_active)
            payload["dot_jvp_update_inactive"] = dict(self.dot_jvp_update_inactive)
            payload["raw_dot_rep_active"] = dict(self.raw_dot_rep_active)
            payload["raw_dot_rep_inactive"] = dict(self.raw_dot_rep_inactive)
            payload["raw_dot_diff_active"] = dict(self.raw_dot_diff_active)
            payload["raw_dot_diff_inactive"] = dict(self.raw_dot_diff_inactive)
            payload["raw_dot_deriv_active"] = dict(self.raw_dot_deriv_active)
            payload["raw_dot_deriv_inactive"] = dict(self.raw_dot_deriv_inactive)
            payload["raw_dot_deriv10_active"] = dict(self.raw_dot_deriv10_active)
            payload["raw_dot_deriv10_inactive"] = dict(self.raw_dot_deriv10_inactive)
            payload["raw_dot_deriv50_active"] = dict(self.raw_dot_deriv50_active)
            payload["raw_dot_deriv50_inactive"] = dict(self.raw_dot_deriv50_inactive)
            payload["raw_dot_loss_grad_active"] = dict(self.raw_dot_loss_grad_active)
            payload["raw_dot_loss_grad_inactive"] = dict(self.raw_dot_loss_grad_inactive)
            payload["raw_dot_loss_grad_active_std"] = dict(self.raw_dot_loss_grad_active_std)
            payload["raw_dot_loss_grad_inactive_std"] = dict(self.raw_dot_loss_grad_inactive_std)
            payload["lr_cell"] = dict(self.lr_cell)
            payload["W_lr_final"] = dict(self.W_lr_final)
            payload["dot_lr_rep_active"] = dict(self.dot_lr_rep_active)
            payload["dot_lr_rep_inactive"] = dict(self.dot_lr_rep_inactive)
            payload["dot_lr_diff_active"] = dict(self.dot_lr_diff_active)
            payload["dot_lr_diff_inactive"] = dict(self.dot_lr_diff_inactive)
            payload["dot_lr_deriv_active"] = dict(self.dot_lr_deriv_active)
            payload["dot_lr_deriv_inactive"] = dict(self.dot_lr_deriv_inactive)
            payload["dot_lr_deriv10_active"] = dict(self.dot_lr_deriv10_active)
            payload["dot_lr_deriv10_inactive"] = dict(self.dot_lr_deriv10_inactive)
            payload["dot_lr_deriv50_active"] = dict(self.dot_lr_deriv50_active)
            payload["dot_lr_deriv50_inactive"] = dict(self.dot_lr_deriv50_inactive)
            payload["dot_lr_loss_grad_active"] = dict(self.dot_lr_loss_grad_active)
            payload["dot_lr_loss_grad_inactive"] = dict(self.dot_lr_loss_grad_inactive)
            payload["dot_lr_loss_grad_active_std"] = dict(self.dot_lr_loss_grad_active_std)
            payload["dot_lr_loss_grad_inactive_std"] = dict(self.dot_lr_loss_grad_inactive_std)
            payload["raw_dot_lr_rep_active"] = dict(self.raw_dot_lr_rep_active)
            payload["raw_dot_lr_rep_inactive"] = dict(self.raw_dot_lr_rep_inactive)
            payload["raw_dot_lr_diff_active"] = dict(self.raw_dot_lr_diff_active)
            payload["raw_dot_lr_diff_inactive"] = dict(self.raw_dot_lr_diff_inactive)
            payload["raw_dot_lr_deriv_active"] = dict(self.raw_dot_lr_deriv_active)
            payload["raw_dot_lr_deriv_inactive"] = dict(self.raw_dot_lr_deriv_inactive)
            payload["raw_dot_lr_deriv10_active"] = dict(self.raw_dot_lr_deriv10_active)
            payload["raw_dot_lr_deriv10_inactive"] = dict(self.raw_dot_lr_deriv10_inactive)
            payload["raw_dot_lr_deriv50_active"] = dict(self.raw_dot_lr_deriv50_active)
            payload["raw_dot_lr_deriv50_inactive"] = dict(self.raw_dot_lr_deriv50_inactive)
            payload["raw_dot_lr_loss_grad_active"] = dict(self.raw_dot_lr_loss_grad_active)
            payload["raw_dot_lr_loss_grad_inactive"] = dict(self.raw_dot_lr_loss_grad_inactive)
            payload["raw_dot_lr_loss_grad_active_std"] = dict(self.raw_dot_lr_loss_grad_active_std)
            payload["raw_dot_lr_loss_grad_inactive_std"] = dict(self.raw_dot_lr_loss_grad_inactive_std)
            payload["dot_lr_ntk_grad_pair"] = {
                combo: dict(self.dot_lr_ntk_grad_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["raw_dot_lr_ntk_grad_pair"] = {
                combo: dict(self.raw_dot_lr_ntk_grad_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["dot_lr_ntk_pair"] = {
                combo: dict(self.dot_lr_ntk_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["raw_dot_lr_ntk_pair"] = {
                combo: dict(self.raw_dot_lr_ntk_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["final_dom_top1_curve"] = {
                name: dict(self.final_dom_top1_curve[name])
                for name in ("rep", "diff", "deriv", "deriv10", "deriv50",
                             "loss_grad")
            }
            payload["final_lr_top1_curve"] = {
                name: dict(self.final_lr_top1_curve[name])
                for name in ("rep", "diff", "deriv", "deriv10", "deriv50",
                             "loss_grad")
            }
            payload["raw_dot_jvp_update_active"] = dict(self.raw_dot_jvp_update_active)
            payload["raw_dot_jvp_update_inactive"] = dict(self.raw_dot_jvp_update_inactive)
            payload["dot_ntk_pair"] = {
                combo: dict(self.dot_ntk_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["raw_dot_ntk_pair"] = {
                combo: dict(self.raw_dot_ntk_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["dot_ntk_grad_pair"] = {
                combo: dict(self.dot_ntk_grad_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["raw_dot_ntk_grad_pair"] = {
                combo: dict(self.raw_dot_ntk_grad_pair[combo])
                for combo in ("ta_sa", "ta_si", "ti_sa", "ti_si")
            }
            payload["norm_diff_active"] = dict(self.norm_diff_active)
            payload["norm_diff_inactive"] = dict(self.norm_diff_inactive)
            payload["norm_deriv_active"] = dict(self.norm_deriv_active)
            payload["norm_deriv_inactive"] = dict(self.norm_deriv_inactive)
            payload["norm_deriv10_active"] = dict(self.norm_deriv10_active)
            payload["norm_deriv10_inactive"] = dict(self.norm_deriv10_inactive)
            payload["norm_deriv50_active"] = dict(self.norm_deriv50_active)
            payload["norm_deriv50_inactive"] = dict(self.norm_deriv50_inactive)
            payload["final_top1_rep"] = {ell: v.cpu()
                                         for ell, v in self.final_top1_rep.items()}
        if any(self.dot_probe_active[frac] for frac in self.cfg.probe_grad_active_fracs):
            # cos(w_c^final, δh_c_probe(t)): probe-batch virtual-grad deltas
            # (class-balanced, denoised). Shape (T, D, L, K) per target layer.
            payload["dot_probe_active"] = {
                frac: dict(self.dot_probe_active[frac])
                for frac in self.cfg.probe_grad_active_fracs
            }
            payload["dot_probe_inactive"] = {
                frac: dict(self.dot_probe_inactive[frac])
                for frac in self.cfg.probe_grad_active_fracs
            }
            payload["cfg"]["probe_grad_batch_size"] = self.cfg.probe_grad_batch_size
            payload["cfg"]["probe_grad_active_fracs"] = self.cfg.probe_grad_active_fracs
            payload["cfg"]["probe_grad_lr"] = self.cfg.probe_grad_lr
        if self.cfg.steer_n_classes > 0:
            payload["steer"] = {ell: torch.stack(v, dim=0)          # (T, Ks)
                                for ell, v in self.history["steer"].items()}
            payload["steer_baseline"] = {ell: torch.stack(v, dim=0)
                                         for ell, v in self.history["steer_baseline"].items()}
            payload["steer_cell"] = {ell: torch.stack(v, dim=0)     # (T, 2)
                                     for ell, v in self.history["steer_cell"].items()}
            payload["steer_targets"] = {ell: v.cpu()
                                        for ell, v in self.steer_targets.items()}
        if self.cfg.mlp_hidden > 0:
            payload["mlp_top1"] = {ell: torch.stack(v, dim=0)       # (T, D, L)
                                   for ell, v in self.history["mlp_top1"].items()}
            payload["mlp_top5"] = {ell: torch.stack(v, dim=0)
                                   for ell, v in self.history["mlp_top5"].items()}
            payload["cfg"]["mlp_hidden"] = self.cfg.mlp_hidden
            payload["cfg"]["mlp_iters"] = self.cfg.mlp_iters
            payload["cfg"]["mlp_lr"] = self.cfg.mlp_lr
            payload["cfg"]["mlp_activation"] = self.cfg.mlp_activation
        if self.cfg.logreg_iters > 0:
            payload["logreg_top1"] = {ell: torch.stack(v, dim=0)
                                      for ell, v in self.history["logreg_top1"].items()}
            payload["logreg_top5"] = {ell: torch.stack(v, dim=0)
                                      for ell, v in self.history["logreg_top5"].items()}
            payload["cfg"]["logreg_iters"] = self.cfg.logreg_iters
            payload["cfg"]["logreg_lr"] = self.cfg.logreg_lr
        torch.save(payload, path)
