"""Central configs. Plain dataclasses — pass them around or override fields."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GraphConfig:
    n_layers: int = 5
    nodes_per_layer: Tuple[int, ...] = (1, 100, 100, 100, 100)
    edges_per_node: int = 10
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.nodes_per_layer, tuple):
            self.nodes_per_layer = tuple(self.nodes_per_layer)
        assert len(self.nodes_per_layer) == self.n_layers, (
            f"len(nodes_per_layer)={len(self.nodes_per_layer)} != n_layers={self.n_layers}"
        )
        for i in range(self.n_layers - 1):
            assert self.edges_per_node <= self.nodes_per_layer[i + 1], (
                f"edges_per_node={self.edges_per_node} exceeds nodes_per_layer[{i+1}]"
                f"={self.nodes_per_layer[i+1]}; sampling without replacement is impossible."
            )


@dataclass
class ModelConfig:
    vocab_size: int = 10        # = edges_per_node
    seq_len: int = 4            # = n_layers - 1
    n_classes: int = 100        # = nodes in the final layer
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_blocks: int = 4
    dropout: float = 0.0
    pos_enc: str = "learned"    # "learned" | "sinusoidal"
    causal: bool = True
    use_residual: bool = True   # if False, transformer blocks drop the skip connection
    use_mlp: bool = True        # if False, blocks are attention-only
    mlp_type: str = "standard"     # "standard" (Linear-act-Linear) | "bilinear" (Linear-x-Linear-then-Linear)
    mlp_activation: str = "gelu"   # "gelu" | "relu" | "relu2" | "tanh" (ignored when mlp_type="bilinear")
    init_std: float = 0.02          # stddev for normal init of embeddings, linears, and bilinear head factors
    suo_init: bool = False         # if True, re-init block attn+MLP linears as semi-orthogonal
    # Replace attention's V linear projection with a 2-layer MLP (same activation as block MLPs).
    attn_v_mlp: bool = False
    attn_v_mlp_hidden: int = -1    # -1 -> d_model
    freeze_attn_v_mlp: bool = False  # if True, V MLPs are random fixed nonlinear projections
    # Same idea for Q and K: replace their linear projections with 2-layer MLPs (per head).
    attn_qk_mlp: bool = False
    attn_qk_mlp_hidden: int = -1
    freeze_attn_qk_mlp: bool = False
    # Classification head: "linear" (default) or "bilinear" (logits[c] = h.T @ U_c @ V_c.T @ h).
    head_type: str = "linear"
    head_rank: int = 8             # only used by bilinear head
    # If True, replace both ln_f and head with nn.Identity so logits = h[:, -1].
    # Requires d_model == n_classes.
    identity_unembed: bool = False
    # If True, initialize the linear unembedding semi-orthogonally and freeze it.
    frozen_suo_unembed: bool = False
    # Frozen MLP on top of the transformer, between ln_f and the trainable head.
    frozen_mlp_layers: int = 0      # 0 disables the MLP
    frozen_mlp_hidden: int = -1     # -1 -> d_model
    frozen_mlp_activation: str = "relu2"   # "relu2" | "relu" | "gelu" | "tanh"


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 0.1   # applied only to 2D weights (matrices/embeddings)
    betas: Tuple[float, float] = (0.9, 0.95)
    n_epochs: int = 200
    # If > 0, compute the cosine warmup+decay schedule as if we were training
    # for `lr_schedule_epochs` epochs, but stop after `n_epochs`. Useful for
    # ablations that want a schedule "borrowed" from a longer reference run.
    lr_schedule_epochs: int = 0
    warmup_frac: float = 0.05
    seed: int = 42
    train_frac: float = 0.8
    device: str = "cuda"
    eval_every_epochs: int = 1
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    optimizer: str = "adamw"      # "adamw" | "sgd" | "klshampoo"
    # Training loss: "ce" (cross-entropy, default) or "mse" (mean squared error
    # against one-hot targets - no softmax anywhere). MSE is used to test whether
    # softmax's exp is what shapes h into a linearly-DoM-decodable form.
    loss_type: str = "ce"
    ckpt_every_epochs: int = 0    # save intermediate ckpts to out_dir/ckpts/ep{N}.pt (0 = only final)
    ckpt_at_steps: Tuple[int, ...] = ()  # save ckpt to out_dir/ckpt_step{N}.pt at each target step
    # If >= 0, each real training batch is stratified by nodes[:, layer].
    # Example: layer=4, 100 latents, batch=8000 -> 80 samples per latent.
    balance_train_layer: int = -1
    # Adversarial linear probes at every (depth, position) for INTERMEDIATE graph layers.
    # Gradient reversal pushes the model to make h less linearly decodable while
    # the probes are trained jointly to recover latents. 0.0 disables.
    adv_probe_lambda: float = 0.0
    # Soft (semi-)orthogonality penalty on every nn.Linear weight in the model:
    # sum ||W^T W - I||_F^2 (or ||W W^T - I||_F^2 if it's fatter than tall).
    # Pushes weights toward Stiefel manifold. 0.0 disables.
    ortho_lambda: float = 0.0
    # Closed-form ridge adversary: at each (depth, position, intermediate
    # graph layer), penalize the per-batch optimal ridge fit quality. Forces
    # latents to be non-linearly-decodable. No learnable probe parameters.
    ridge_adv_lambda: float = 0.0
    ridge_adv_nu: float = 1e-3       # ridge regularizer (scaled by diag(X^T X) mean)
    ridge_adv_tau: float = 0.05      # logsumexp temperature; smaller -> closer to max-pool
    # Difference-of-means probe tracker: log DoM-probe AUC every K steps for a
    # fixed set of sampled node latents at each intermediate graph layer. Cheap;
    # produces a (T, D, L, K) AUC cube per latent layer saved as dom_probe.pt.
    dom_probe_every_steps: int = 0   # 0 disables
    dom_probe_n_nodes: int = 20
    dom_probe_seed: int = 0
    # Cap the probe eval subset. Each probe-step forward pass runs on this many
    # examples, and finalize_dot_products iterates history at this size, so
    # dropping these is the biggest lever for probe-time cost.
    dom_probe_max_train: int = 60000
    dom_probe_max_test: int = 5000
    # Steering intervention (piggybacks on the DoM probe). At each probe step,
    # at each layer's currently-best DoM cell, add alpha * (mu+ - mu-) to the
    # residual for K target classes and measure fraction of predictions landing
    # on a graph-6 node reachable from the target class. 0 disables.
    dom_steer_n_classes: int = 0
    dom_steer_alpha: float = 1.0
    # 1-hidden-layer MLP probes fit at every (d, p) cell per probed step.
    # 0 disables. Adds only ~1-2 min to a 10-min run thanks to batched bmm.
    dom_mlp_hidden: int = 0
    dom_mlp_iters: int = 100
    dom_mlp_lr: float = 3e-2
    dom_mlp_activation: str = "relu"
    # Multinomial logistic regression probes at every (d, p) cell. 0 disables.
    dom_logreg_iters: int = 0
    dom_logreg_lr: float = 3e-2
    # Class-balanced probe-batch virtual-grad delta_h analysis: at each probe
    # step, take one SGD step on fixed per-class batches with these positive
    # fractions and record cos(w_c^final, delta_h) for the gradient plot.
    dom_probe_grad_batch_size: int = 200
    dom_probe_grad_active_fracs: Tuple[float, ...] = (0.01, 0.1, 0.5)
    dom_probe_grad_lr: float = 0.0     # 0 -> use train_cfg.lr
    # Differentiable DoM erasure inside the forward pass. For K sampled classes
    # at graph layer `erase_layer`, compute the batch DoM direction at every
    # (depth, position) and project it out of the residual stream. Because DoM
    # is a function of the batch's hidden states, gradients flow through the
    # projection: the model can only reduce loss by making the target latents
    # actually invisible in the residual stream. -1 disables.
    erase_layer: int = -1
    erase_n_features: int = 0
    erase_seed: int = 0
    erase_ridge: float = 1e-4

    def __post_init__(self):
        if not isinstance(self.betas, tuple):
            self.betas = tuple(self.betas)
        if not isinstance(self.ckpt_at_steps, tuple):
            self.ckpt_at_steps = tuple(self.ckpt_at_steps)
        if not isinstance(self.dom_probe_grad_active_fracs, tuple):
            self.dom_probe_grad_active_fracs = tuple(self.dom_probe_grad_active_fracs)


@dataclass
class ProbeConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-4
    n_epochs: int = 200
    batch_size: int = 1024
    device: str = "cuda"
    skip_trivial_layer0: bool = True   # layer 0 has a single class
