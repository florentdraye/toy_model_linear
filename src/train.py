"""Training loop: GPU-resident, no DataLoader, no per-step host sync.

Data lives on the GPU for the whole run; each step indexes into it. Loss/acc
are accumulated in device tensors and only synced to host once per eval.
This makes overhead negligible relative to the forward/backward pass.
"""
import math
import time
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD

from .config import TrainConfig, ModelConfig, GraphConfig
from .model import ToyTransformer
from .graph import Graph
from .data import enumerate_paths, split_indices
from .adv_probe import AdversarialProbes
from .ridge_adv import ridge_adv_loss
from .dom_probe import DoMProbeConfig, DoMProbeTracker


def make_optimizer(model: nn.Module, train_cfg: TrainConfig):
    if train_cfg.optimizer == "adamw":
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(groups, lr=train_cfg.lr, betas=train_cfg.betas, fused=True)
    if train_cfg.optimizer == "sgd":
        params = [p for p in model.parameters() if p.requires_grad]
        return SGD(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    if train_cfg.optimizer == "klshampoo":
        from .kl_shampoo import KLShampoo
        params = [p for p in model.parameters() if p.requires_grad]
        return KLShampoo(
            params,
            lr=train_cfg.lr,
            betas=train_cfg.betas,
            weight_decay=train_cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer {train_cfg.optimizer!r}")


def _optimizer_theta_dot(model: nn.Module, optimizer: torch.optim.Optimizer,
                         train_cfg: TrainConfig) -> dict[str, torch.Tensor]:
    group_by_param = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_by_param[id(p)] = group
    theta_dot = {}
    for name, p in model.named_parameters():
        group = group_by_param.get(id(p))
        if group is None or p.grad is None:
            theta_dot[name] = torch.zeros_like(p)
            continue
        grad = p.grad.detach()
        lr = group["lr"]
        weight_decay = group.get("weight_decay", 0.0)
        if train_cfg.optimizer == "sgd":
            update_grad = grad
            if weight_decay:
                update_grad = update_grad + weight_decay * p.detach()
            theta_dot[name] = -lr * update_grad
        elif train_cfg.optimizer == "adamw":
            beta1, beta2 = group.get("betas", train_cfg.betas)
            eps = group.get("eps", 1e-8)
            state = optimizer.state[p]
            step_t = int(state.get("step", 0)) + 1
            exp_avg = state.get("exp_avg", torch.zeros_like(p))
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(p))
            m = beta1 * exp_avg + (1.0 - beta1) * grad
            v = beta2 * exp_avg_sq + (1.0 - beta2) * grad.square()
            m_hat = m / (1.0 - beta1 ** step_t)
            v_hat = v / (1.0 - beta2 ** step_t)
            update = m_hat / (v_hat.sqrt() + eps)
            if weight_decay:
                update = update + weight_decay * p.detach()
            theta_dot[name] = -lr * update
        else:
            theta_dot[name] = torch.zeros_like(p)
    return theta_dot


def ortho_penalty(model: nn.Module) -> torch.Tensor:
    """sum_W ||G_W - I||_F^2  where G_W = W W^T if W is wide else W^T W.

    Walks every nn.Linear with a trainable weight. Penalizes both unit-norm
    deviation AND off-diagonal Gram mass on the smaller dimension — the
    classical "soft semi-orthogonality" regularizer.
    """
    total = None
    for m in model.modules():
        if not isinstance(m, nn.Linear) or not m.weight.requires_grad:
            continue
        W = m.weight
        G = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        term = (G - I).pow(2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _make_balanced_class_indices(nodes_train: torch.Tensor, layer: int
                                 ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if layer < 0 or layer >= nodes_train.shape[1]:
        raise ValueError(
            f"balance_train_layer={layer} outside node columns [0, {nodes_train.shape[1]})"
        )
    labels = nodes_train[:, layer]
    classes = labels.unique(sorted=True)
    class_indices = []
    for c in classes:
        idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        if idx.numel() == 0:
            raise ValueError(f"no training examples for layer {layer} class {int(c.item())}")
        class_indices.append(idx)
    return classes, class_indices


def _balanced_batch_indices(class_indices: list[torch.Tensor], batch_size: int,
                            step: int, device: str) -> torch.Tensor:
    n_classes = len(class_indices)
    if batch_size < n_classes:
        raise ValueError(
            f"batch_size={batch_size} is smaller than {n_classes} balance classes"
        )
    base = batch_size // n_classes
    extra = batch_size % n_classes
    pieces = []
    for j, idxs in enumerate(class_indices):
        n = base + (1 if ((j - step) % n_classes) < extra else 0)
        picks = torch.randint(idxs.numel(), (n,), device=device)
        pieces.append(idxs[picks])
    batch = torch.cat(pieces)
    return batch[torch.randperm(batch.numel(), device=device)]


@torch.no_grad()
def evaluate_tensors(model: nn.Module, edges: torch.Tensor, labels: torch.Tensor,
                     batch_size: int, amp_dtype=torch.bfloat16,
                     loss_type: str = "ce", n_classes: int | None = None) -> tuple:
    model.eval()
    n = edges.shape[0]
    loss_sum = torch.zeros((), device=edges.device)
    correct = torch.zeros((), device=edges.device)
    for i in range(0, n, batch_size):
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(edges[i:i + batch_size])
        y = labels[i:i + batch_size]
        if loss_type == "ce":
            loss_sum += F.cross_entropy(logits.float(), y, reduction="sum")
        else:
            oh = F.one_hot(y, n_classes).float()
            loss_sum += F.mse_loss(logits.float(), oh, reduction="none").mean(-1).sum()
        correct += (logits.argmax(-1) == y).sum()
    return (loss_sum / n).item(), (correct / n).item()


def _sync_model_to_graph(model_cfg: ModelConfig, graph_cfg: GraphConfig) -> ModelConfig:
    fields = asdict(model_cfg)
    want = {
        "seq_len": graph_cfg.n_layers - 1,
        "vocab_size": graph_cfg.edges_per_node,
        "n_classes": graph_cfg.nodes_per_layer[-1],
    }
    changed = {k: (fields[k], v) for k, v in want.items() if fields[k] != v}
    if changed:
        for k, (old, new) in changed.items():
            print(f"[sync] model_cfg.{k}: {old} -> {new}")
        fields.update(want)
        return ModelConfig(**fields)
    return model_cfg


def train(graph_cfg: GraphConfig, model_cfg: ModelConfig, train_cfg: TrainConfig,
          out_dir: str = "runs/default",
          *,
          init_state_dict: dict | None = None,
          init_split: tuple | None = None,
          init_graph: Graph | None = None) -> dict:
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device
    torch.set_float32_matmul_precision("high")  # enable TF32 matmul on Ampere+

    graph = init_graph if init_graph is not None else Graph(graph_cfg)
    paths = enumerate_paths(graph)
    edges_all = paths["edge_seqs"].to(device)
    nodes_all = paths["nodes"].to(device)
    labels_all = nodes_all[:, -1]

    N = edges_all.shape[0]
    if init_split is not None:
        train_idx_cpu, test_idx_cpu = init_split
    else:
        train_idx_cpu, test_idx_cpu = split_indices(N, train_cfg.train_frac, train_cfg.seed)
    train_idx = train_idx_cpu.to(device)
    test_idx = test_idx_cpu.to(device)

    edges_train = edges_all[train_idx]
    labels_train = labels_all[train_idx]
    nodes_train = nodes_all[train_idx]
    edges_test = edges_all[test_idx]
    labels_test = labels_all[test_idx]

    model_cfg = _sync_model_to_graph(model_cfg, graph_cfg)
    model = ToyTransformer(model_cfg).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
        print(f"[init] loaded model state from checkpoint")

    adv_probes = None
    if train_cfg.adv_probe_lambda > 0:
        adv_probes = AdversarialProbes(
            d_model=model_cfg.d_model,
            n_blocks=model_cfg.n_blocks,
            seq_len=model_cfg.seq_len,
            nodes_per_layer=list(graph_cfg.nodes_per_layer),
            n_layers=graph_cfg.n_layers,
        ).to(device)

    # Intermediate graph layers attacked by the ridge adversary (mirrors AdversarialProbes).
    ridge_attack_layers = [
        ell for ell in range(1, graph_cfg.n_layers - 1)
        if graph_cfg.nodes_per_layer[ell] > 1
    ]
    use_ridge_adv = train_cfg.ridge_adv_lambda > 0
    need_hidden = (adv_probes is not None) or use_ridge_adv

    optimizer = make_optimizer(model, train_cfg)
    if adv_probes is not None:
        # Probes train normally (no decay, regardless of optimizer choice).
        optimizer.add_param_group({"params": list(adv_probes.parameters()),
                                   "weight_decay": 0.0})

    n_train = edges_train.shape[0]
    n_test = edges_test.shape[0]
    steps_per_epoch = math.ceil(n_train / train_cfg.batch_size)
    total_steps = train_cfg.n_epochs * steps_per_epoch
    sched_epochs = (train_cfg.lr_schedule_epochs
                    if train_cfg.lr_schedule_epochs > 0
                    else train_cfg.n_epochs)
    sched_total_steps = sched_epochs * steps_per_epoch
    warmup_steps = max(1, int(train_cfg.warmup_frac * sched_total_steps))
    eval_batch = max(train_cfg.batch_size, 8192)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph.save(out_dir / "graph.pt")
    ckpt_steps_set = set(train_cfg.ckpt_at_steps)

    log = {"epoch": [], "step": [], "train_loss": [], "train_acc": [],
           "test_loss": [], "test_acc": [], "lr": [], "ortho": [], "ridge": []}

    balanced_classes = None
    balanced_class_indices = None
    if train_cfg.balance_train_layer >= 0:
        balanced_classes, balanced_class_indices = _make_balanced_class_indices(
            nodes_train, train_cfg.balance_train_layer
        )

    erase_targets = None
    if train_cfg.erase_layer >= 0 and train_cfg.erase_n_features > 0:
        ell = train_cfg.erase_layer
        if ell >= graph_cfg.n_layers:
            raise ValueError(f"erase_layer={ell} outside [0, {graph_cfg.n_layers})")
        reachable = nodes_train[:, ell].unique()
        K = min(train_cfg.erase_n_features, reachable.numel())
        gen = torch.Generator(device="cpu").manual_seed(train_cfg.erase_seed)
        perm = torch.randperm(reachable.numel(), generator=gen)[:K]
        erase_targets = reachable[perm.to(device)]
        print(f"[erase] projecting DoM(layer={ell}) for {K} sampled classes "
              f"at every residual depth/position, ids={erase_targets.tolist()}",
              flush=True)

    dom_tracker = None
    if train_cfg.dom_probe_every_steps > 0:
        dom_tracker = DoMProbeTracker(
            edges=edges_all, nodes=nodes_all,
            train_idx=train_idx, test_idx=test_idx,
            n_layers=graph_cfg.n_layers,
            nodes_per_layer=list(graph_cfg.nodes_per_layer),
            cfg=DoMProbeConfig(
                n_nodes=train_cfg.dom_probe_n_nodes,
                every_steps=train_cfg.dom_probe_every_steps,
                sample_seed=train_cfg.dom_probe_seed,
                probe_max_train=train_cfg.dom_probe_max_train,
                probe_max_test=train_cfg.dom_probe_max_test,
                steer_n_classes=train_cfg.dom_steer_n_classes,
                steer_alpha=train_cfg.dom_steer_alpha,
                mlp_hidden=train_cfg.dom_mlp_hidden,
                mlp_iters=train_cfg.dom_mlp_iters,
                mlp_lr=train_cfg.dom_mlp_lr,
                mlp_activation=train_cfg.dom_mlp_activation,
                logreg_iters=train_cfg.dom_logreg_iters,
                logreg_lr=train_cfg.dom_logreg_lr,
                probe_grad_batch_size=train_cfg.dom_probe_grad_batch_size,
                probe_grad_active_fracs=train_cfg.dom_probe_grad_active_fracs,
                probe_grad_lr=(train_cfg.dom_probe_grad_lr
                               if train_cfg.dom_probe_grad_lr > 0
                               else train_cfg.lr),
            ),
        )
        dom_tracker.run(model, step=0)
        print(f"[dom-probe] tracking layers {dom_tracker.target_layers} "
              f"with {train_cfg.dom_probe_n_nodes} nodes each, "
              f"every {train_cfg.dom_probe_every_steps} steps", flush=True)

    print(f"#params:           {model.num_params():,}")
    print(f"#train / #test:    {n_train} / {n_test}")
    if sched_total_steps != total_steps:
        print(f"#steps (warmup):   {total_steps} ({warmup_steps})   "
              f"[LR schedule spans {sched_total_steps} steps]")
    else:
        print(f"#steps (warmup):   {total_steps} ({warmup_steps})")
    print(f"weight decay:      {train_cfg.weight_decay}")
    print(f"device:            {device}  batch={train_cfg.batch_size}")
    if balanced_classes is not None:
        per_class = train_cfg.batch_size // balanced_classes.numel()
        rem = train_cfg.batch_size % balanced_classes.numel()
        print(f"batch sampler:     balanced layer {train_cfg.balance_train_layer} "
              f"over {balanced_classes.numel()} latents "
              f"({per_class} each + {rem} rotating extras)")

    t0 = time.time()
    step = 0
    for epoch in range(train_cfg.n_epochs):
        model.train()
        perm = None
        if balanced_class_indices is None:
            perm = torch.randperm(n_train, device=device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        ridge_sum = torch.zeros((), device=device)
        ridge_count = 0
        seen = 0

        for batch_step in range(steps_per_epoch):
            if balanced_class_indices is None:
                i = batch_step * train_cfg.batch_size
                idx = perm[i:i + train_cfg.batch_size]
            else:
                idx = _balanced_batch_indices(
                    balanced_class_indices, train_cfg.batch_size, step, device
                )
            edges = edges_train[idx]
            labels = labels_train[idx]

            lr_scale = cosine_warmup(step, sched_total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = train_cfg.lr * lr_scale

            erase_kwargs = None
            if erase_targets is not None:
                erase_kwargs = {
                    "labels": nodes_train[idx, train_cfg.erase_layer],
                    "targets": erase_targets,
                    "ridge": train_cfg.erase_ridge,
                }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if need_hidden:
                    logits, hiddens = model(edges, return_hidden=True,
                                            erase=erase_kwargs)
                else:
                    logits = model(edges, erase=erase_kwargs)
                if train_cfg.loss_type == "ce":
                    main_loss = F.cross_entropy(logits, labels,
                                                label_smoothing=train_cfg.label_smoothing)
                else:
                    oh = F.one_hot(labels, model_cfg.n_classes).float()
                    main_loss = F.mse_loss(logits, oh)
                if adv_probes is not None:
                    nodes_batch = nodes_train[idx]
                    probe_loss = adv_probes.loss(hiddens, nodes_batch,
                                                 train_cfg.adv_probe_lambda)
                    loss = main_loss + probe_loss
                else:
                    loss = main_loss
            if train_cfg.ortho_lambda > 0:
                ortho = ortho_penalty(model)
                loss = loss + train_cfg.ortho_lambda * ortho
            if use_ridge_adv:
                nodes_batch = nodes_train[idx]
                ridge_pen = ridge_adv_loss(
                    hiddens, nodes_batch, ridge_attack_layers,
                    list(graph_cfg.nodes_per_layer),
                    nu=train_cfg.ridge_adv_nu, tau=train_cfg.ridge_adv_tau,
                )
                loss = loss + train_cfg.ridge_adv_lambda * ridge_pen
                ridge_sum = ridge_sum + ridge_pen.detach()
                ridge_count += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                loss_sum += main_loss.detach() * labels.numel()
                correct += (logits.argmax(-1) == labels).sum()
                seen += labels.numel()
            step += 1
            if dom_tracker is not None:
                dom_tracker.tick(model, step=step)
            if step in ckpt_steps_set:
                torch.save({
                    "model_state": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "graph_cfg": asdict(graph_cfg),
                    "train_cfg": asdict(train_cfg),
                    "split": {"train": train_idx_cpu, "test": test_idx_cpu},
                    "step": step,
                }, out_dir / f"ckpt_step{step}.pt")
                print(f"[ckpt] saved out_dir/ckpt_step{step}.pt", flush=True)

        if (epoch + 1) % train_cfg.eval_every_epochs == 0 or epoch == train_cfg.n_epochs - 1:
            train_loss = (loss_sum / seen).item()
            train_acc = (correct / seen).item()
            test_loss, test_acc = evaluate_tensors(
                model, edges_test, labels_test, eval_batch,
                loss_type=train_cfg.loss_type, n_classes=model_cfg.n_classes,
            )
            with torch.no_grad():
                ortho_val = ortho_penalty(model).item() if train_cfg.ortho_lambda > 0 else 0.0
            ridge_val = (ridge_sum / max(ridge_count, 1)).item() if use_ridge_adv else 0.0
            dt = time.time() - t0
            print(
                f"ep {epoch+1:4d}/{train_cfg.n_epochs}  "
                f"step {step:6d}  "
                f"train {train_loss:.4f}/{train_acc:.3f}  "
                f"test {test_loss:.4f}/{test_acc:.3f}  "
                f"gap {train_acc - test_acc:+.3f}  "
                f"ortho {ortho_val:.2f}  "
                f"ridge {ridge_val:.3f}  "
                f"lr {train_cfg.lr * lr_scale:.2e}  "
                f"({dt:.1f}s)",
                flush=True,
            )
            log["epoch"].append(epoch + 1)
            log["step"].append(step)
            log["train_loss"].append(train_loss)
            log["train_acc"].append(train_acc)
            log["test_loss"].append(test_loss)
            log["test_acc"].append(test_acc)
            log["lr"].append(train_cfg.lr * lr_scale)
            log["ortho"].append(ortho_val)
            log["ridge"].append(ridge_val)

    if dom_tracker is not None:
        dom_tracker.finalize_dot_products(model)
        dom_tracker.save(out_dir / "dom_probe.pt")
        print(f"saved dom probe log -> {out_dir / 'dom_probe.pt'}", flush=True)

    ckpt_path = out_dir / "ckpt.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "graph_cfg": asdict(graph_cfg),
        "train_cfg": asdict(train_cfg),
        "split": {"train": train_idx_cpu, "test": test_idx_cpu},
        "log": log,
    }, ckpt_path)
    print(f"saved ckpt -> {ckpt_path}  ({time.time()-t0:.1f}s total)", flush=True)

    return {"model": model, "graph": graph, "log": log,
            "split": (train_idx_cpu, test_idx_cpu),
            "out_dir": out_dir, "model_cfg": model_cfg}
