"""Minimal hackable transformer for sequence classification.

Pre-LN blocks, learned (or sinusoidal) positional embeddings, causal MHA.
Classification head reads the last position. Hand-written so swapping pieces
(e.g. RoPE, MoE, different attn) is straightforward.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with F.scaled_dot_product_attention (FlashAttention on H100).

    If cfg.attn_v_mlp is True, V is computed by a 2-layer MLP instead of a single
    Linear projection. Q and K remain a fused Linear(d_model, 2*d_model).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.use_qk_mlp = cfg.attn_qk_mlp
        self.use_v_mlp = cfg.attn_v_mlp

        if self.use_qk_mlp:
            qk_hidden = cfg.attn_qk_mlp_hidden if cfg.attn_qk_mlp_hidden > 0 else cfg.d_model
            self.q_mlp = _make_mlp(cfg, cfg.d_model, qk_hidden, cfg.d_model)
            self.k_mlp = _make_mlp(cfg, cfg.d_model, qk_hidden, cfg.d_model)
        if self.use_v_mlp:
            v_hidden = cfg.attn_v_mlp_hidden if cfg.attn_v_mlp_hidden > 0 else cfg.d_model
            self.v_mlp = _make_mlp(cfg, cfg.d_model, v_hidden, cfg.d_model)

        # Linear projections for whatever isn't an MLP, fused when possible.
        if not self.use_qk_mlp and not self.use_v_mlp:
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        elif not self.use_qk_mlp:                  # only V is MLP
            self.qk = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=True)
        elif not self.use_v_mlp:                   # only Q,K are MLP
            self.v_lin = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.dropout = cfg.dropout
        self.causal = cfg.causal

    def forward(self, x):
        B, L, D = x.shape
        # Q, K
        if self.use_qk_mlp:
            q = self.q_mlp(x).view(B, L, self.n_heads, self.d_head)
            k = self.k_mlp(x).view(B, L, self.n_heads, self.d_head)
        elif self.use_v_mlp:
            qk = self.qk(x).view(B, L, 2, self.n_heads, self.d_head)
            q, k = qk.unbind(dim=2)
        else:
            qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
        # V
        if self.use_v_mlp:
            v = self.v_mlp(x).view(B, L, self.n_heads, self.d_head)
        elif self.use_qk_mlp:
            v = self.v_lin(x).view(B, L, self.n_heads, self.d_head)
        # else: v already extracted from fused qkv above

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class ReLU2(nn.Module):
    def forward(self, x):
        return F.relu(x).square()


class BilinearMLP(nn.Module):
    """Bilinear MLP (GLU-without-activation): y = out(left(x) * right(x)).

    Each coordinate of the output is a quadratic form in x. No activation
    function: the elementwise product IS the nonlinearity. Useful as an MLP
    replacement when you want every operation in the network to be polynomial
    in the input rather than a smooth nonlinearity.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.left = nn.Linear(d_in, d_hidden)
        self.right = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.out(self.left(x) * self.right(x))


class BilinearHead(nn.Module):
    """Quadratic-form classifier: logits[c] = h.T @ A_c @ h with low-rank A_c = U_c V_c.T.

    No "answer direction" exists; the discriminative signal lives in pairs of
    coordinates of h. Linear probes cannot recover this; the model must use
    multiplicative interactions to separate classes.
    """

    def __init__(self, d_model: int, n_classes: int, rank: int = 8, init_std: float = 0.02):
        super().__init__()
        self.U = nn.Parameter(torch.empty(n_classes, d_model, rank))
        self.V = nn.Parameter(torch.empty(n_classes, d_model, rank))
        nn.init.normal_(self.U, std=init_std)
        nn.init.normal_(self.V, std=init_std)

    def forward(self, h):
        # h: (B, d). Output: (B, C).
        Uh = torch.einsum("bd,cdr->bcr", h, self.U)
        Vh = torch.einsum("bd,cdr->bcr", h, self.V)
        return (Uh * Vh).sum(-1)


@torch.no_grad()
def suo_initialize(rows: int, cols: int, device=None) -> torch.Tensor:
    """Semi-orthogonal init. Returns W of shape (rows, cols) such that
    W @ W.T = I if rows <= cols, else W.T @ W = I. Norm-preserving through
    the linear; crucial for skipless networks.
    """
    target = device if device is not None else torch.device("cpu")
    compute = torch.device("cuda") if torch.cuda.is_available() else target
    if rows <= cols:
        x = torch.randn(rows, cols, device=compute)
        gram = x @ x.T
        eigvals, eigvecs = torch.linalg.eigh(gram)
        inv_sqrt = eigvals.clamp(min=1e-8).rsqrt()
        return ((eigvecs * inv_sqrt) @ eigvecs.T @ x).to(target)
    x = torch.randn(cols, rows, device=compute)
    gram = x @ x.T
    eigvals, eigvecs = torch.linalg.eigh(gram)
    inv_sqrt = eigvals.clamp(min=1e-8).rsqrt()
    return ((eigvecs * inv_sqrt) @ eigvecs.T @ x).T.contiguous().to(target)


@torch.no_grad()
def get_ortho(dim: int, device=None) -> torch.Tensor:
    """Random orthogonal matrix via SVD of randn. Shape (dim, dim)."""
    target = device if device is not None else torch.device("cpu")
    compute = torch.device("cuda") if torch.cuda.is_available() else target
    M = torch.randn(dim, dim, device=compute)
    U, _, _ = torch.linalg.svd(M)
    return U.to(target)


@torch.no_grad()
def _suo_init_mlp(mlp: nn.Module) -> None:
    """SUO-init every Linear inside a standard nn.Sequential MLP or a BilinearMLP."""
    linears = list(mlp.modules()) if isinstance(mlp, BilinearMLP) else list(mlp)
    for layer in linears:
        if isinstance(layer, nn.Linear):
            r, c = layer.weight.shape
            layer.weight.copy_(suo_initialize(r, c, device=layer.weight.device))
            if layer.bias is not None:
                layer.bias.data.zero_()


@torch.no_grad()
def apply_suo_init_to_blocks(model: "ToyTransformer") -> None:
    """Skipless-friendly init, mirrors residual_free_llm's three-part scheme:
       - Per-head ortho QK with W_Q = W_K   -> attention softmax ~ identity at init
       - Per-head ortho VO with W_V = W_O   -> V * O ~ identity per head at init
       - SUO on MLP linears                  -> norm-preserving through MLP
    All biases zeroed so the near-identity property holds at init.
    """
    cfg = model.cfg
    d = cfg.d_model
    H = cfg.n_heads
    hd = d // H
    assert d % H == 0, "d_model must divide n_heads for per-head init"
    for block in model.blocks:
        proj = block.attn.proj
        device = proj.weight.device
        if block.attn.use_qk_mlp:
            # Q and K are MLPs (and V is either MLP or a separate Linear).
            _suo_init_mlp(block.attn.q_mlp)
            _suo_init_mlp(block.attn.k_mlp)
            if block.attn.use_v_mlp:
                _suo_init_mlp(block.attn.v_mlp)
            else:
                lin = block.attn.v_lin
                r, c = lin.weight.shape
                lin.weight.copy_(suo_initialize(r, c, device=device))
                if lin.bias is not None:
                    lin.bias.data.zero_()
            r, c = proj.weight.shape
            proj.weight.copy_(suo_initialize(r, c, device=device))
            if proj.bias is not None:
                proj.bias.data.zero_()
        elif block.attn.use_v_mlp:
            # QK split: shared per-head ortho on the fused (qk) Linear.
            qk_lin = block.attn.qk     # (2d, d)
            for h in range(H):
                ortho = get_ortho(d, device=device)
                qk_mat = ortho[:, :hd]
                s, e = h * hd, (h + 1) * hd
                qk_lin.weight.data[s:e].copy_(qk_mat.T.contiguous())
                qk_lin.weight.data[d + s:d + e].copy_(qk_mat.T.contiguous())
            if qk_lin.bias is not None:
                qk_lin.bias.data.zero_()
            # V is a 2-layer MLP (standard or bilinear) -> SUO all linears
            _suo_init_mlp(block.attn.v_mlp)
            # O: SUO (no per-head V/O pairing anymore since V isn't linear).
            r, c = proj.weight.shape
            proj.weight.copy_(suo_initialize(r, c, device=proj.weight.device))
            if proj.bias is not None:
                proj.bias.data.zero_()
        else:
            qkv = block.attn.qkv       # (3d, d)
            # Per-head: shared ortho for Q and K
            for h in range(H):
                ortho = get_ortho(d, device=device)
                qk = ortho[:, :hd]
                s, e = h * hd, (h + 1) * hd
                qkv.weight.data[s:e].copy_(qk.T.contiguous())
                qkv.weight.data[d + s:d + e].copy_(qk.T.contiguous())
            # Per-head: shared ortho for V and O
            for h in range(H):
                ortho = get_ortho(d, device=device)
                vo = ortho[:, :hd]
                s, e = h * hd, (h + 1) * hd
                qkv.weight.data[2 * d + s:2 * d + e].copy_(vo.T.contiguous())
                proj.weight.data[s:e].copy_(vo.T.contiguous())
            if qkv.bias is not None:
                qkv.bias.data.zero_()
            if proj.bias is not None:
                proj.bias.data.zero_()
        # SUO on MLP (standard or bilinear)
        if block.use_mlp:
            _suo_init_mlp(block.mlp)


def erase_dom_projection(h: torch.Tensor, labels: torch.Tensor,
                         targets: torch.Tensor, ridge: float = 1e-4
                         ) -> torch.Tensor:
    """Project batch DoM directions for K target classes out of the residual stream.

    For each position p, compute W[k, p, :] = mean(h[i, p] | y_i == c_k)
    - mean(h[i, p] | y_i != c_k) from batch stats, then subtract from h the
    orthogonal projection onto span(W[:, p, :]). Fully differentiable through W.

    h:       (B, L, D) — residual stream at one depth
    labels:  (B,)      — class id per example at some graph layer
    targets: (K,)      — class ids to erase
    ridge:   scalar    — added to Gram diagonal to stabilize the solve
    """
    orig_dtype = h.dtype
    # Disable autocast: linalg.solve on CUDA doesn't support bfloat16, and we
    # want a numerically stable Gram solve regardless of the outer AMP context.
    with torch.autocast("cuda", enabled=False):
        h32 = h.float()
        B, L, D = h32.shape
        K = targets.numel()

        Y = (labels.unsqueeze(1) == targets.unsqueeze(0)).float()  # (B, K)
        n_pos = Y.sum(0)
        n_neg = B - n_pos
        valid = ((n_pos > 0) & (n_neg > 0)).float().view(K, 1, 1)  # zero out degenerate rows
        n_pos_safe = n_pos.clamp(min=1.0).view(K, 1, 1)
        n_neg_safe = n_neg.clamp(min=1.0).view(K, 1, 1)

        sum_pos = torch.einsum('bk,bld->kld', Y, h32)              # (K, L, D)
        sum_all = h32.sum(0, keepdim=False)                        # (L, D)
        sum_neg = sum_all.unsqueeze(0) - sum_pos                   # (K, L, D)
        W = (sum_pos / n_pos_safe - sum_neg / n_neg_safe) * valid  # (K, L, D)

        W_LKD = W.transpose(0, 1)                                  # (L, K, D)
        G = torch.matmul(W_LKD, W_LKD.transpose(-2, -1))           # (L, K, K)
        G_reg = G + ridge * torch.eye(K, device=h32.device, dtype=G.dtype).unsqueeze(0)

        scores = torch.einsum('bld,lkd->blk', h32, W_LKD)          # (B, L, K)
        rhs = scores.permute(1, 2, 0)                              # (L, K, B)
        weights = torch.linalg.solve(G_reg, rhs).permute(2, 0, 1)  # (B, L, K)
        proj = torch.einsum('blk,lkd->bld', weights, W_LKD)        # (B, L, D)

        out = h32 - proj

    return out.to(orig_dtype)


def _make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "relu2":
        return ReLU2()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation {name!r}")


def _make_mlp(cfg: ModelConfig, d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    if cfg.mlp_type == "standard":
        return nn.Sequential(
            nn.Linear(d_in, d_hidden),
            _make_activation(cfg.mlp_activation),
            nn.Linear(d_hidden, d_out),
        )
    if cfg.mlp_type == "bilinear":
        return BilinearMLP(d_in, d_hidden, d_out)
    raise ValueError(f"unknown mlp_type {cfg.mlp_type!r}")


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.use_mlp = cfg.use_mlp
        if cfg.use_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = _make_mlp(cfg, cfg.d_model, cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.use_residual = cfg.use_residual

    def forward(self, x):
        a = self.drop(self.attn(self.ln1(x)))
        x = x + a if self.use_residual else a
        if self.use_mlp:
            m = self.drop(self.mlp(self.ln2(x)))
            x = x + m if self.use_residual else m
        return x


class FrozenMLP(nn.Module):
    """Random-init MLP with all parameters frozen.

    n_layers = number of linear layers. Activation is applied between adjacent
    linears (so n_layers linears -> n_layers - 1 activations). Output dim equals
    input dim; the trainable classification head sits afterward.
    """

    def __init__(self, dim: int, hidden: int, n_layers: int, activation: str):
        super().__init__()
        assert n_layers >= 1
        dims = [dim] + [hidden] * (n_layers - 1) + [dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.activation = activation
        for p in self.parameters():
            p.requires_grad = False

    def _act(self, x):
        if self.activation == "relu2":
            return F.relu(x).square()
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        raise ValueError(f"unknown activation {self.activation!r}")

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = self._act(x)
        return x


class ToyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_enc == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
            nn.init.normal_(self.pos_emb, std=cfg.init_std)
        elif cfg.pos_enc == "sinusoidal":
            self.register_buffer(
                "pos_emb", self._sinusoidal(cfg.seq_len, cfg.d_model),
                persistent=False,
            )
        else:
            raise ValueError(f"unknown pos_enc {cfg.pos_enc!r}")

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        if cfg.frozen_mlp_layers > 0:
            hidden = cfg.frozen_mlp_hidden if cfg.frozen_mlp_hidden > 0 else cfg.d_model
            self.frozen_mlp = FrozenMLP(cfg.d_model, hidden,
                                        cfg.frozen_mlp_layers, cfg.frozen_mlp_activation)
        else:
            self.frozen_mlp = None

        if cfg.head_type == "linear":
            self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        elif cfg.head_type == "bilinear":
            self.head = BilinearHead(cfg.d_model, cfg.n_classes,
                                     rank=cfg.head_rank, init_std=cfg.init_std)
        else:
            raise ValueError(f"unknown head_type {cfg.head_type!r}")
        if cfg.identity_unembed:
            assert not cfg.frozen_suo_unembed, (
                "identity_unembed and frozen_suo_unembed are mutually exclusive"
            )
            assert cfg.d_model == cfg.n_classes, (
                f"identity_unembed requires d_model ({cfg.d_model}) == n_classes ({cfg.n_classes})"
            )
            self.ln_f = nn.Identity()
            self.head = nn.Identity()
        self.apply(lambda m: self._init_weights(m, cfg.init_std))
        if cfg.suo_init:
            apply_suo_init_to_blocks(self)
        if cfg.frozen_suo_unembed:
            assert isinstance(self.head, nn.Linear), (
                "frozen_suo_unembed requires head_type='linear'"
            )
            with torch.no_grad():
                rows, cols = self.head.weight.shape
                self.head.weight.copy_(
                    suo_initialize(rows, cols, device=self.head.weight.device)
                )
                if self.head.bias is not None:
                    self.head.bias.zero_()
            for p in self.head.parameters():
                p.requires_grad = False
        # Freeze the V MLPs (random nonlinear projection inside each attention block)
        if cfg.attn_v_mlp and cfg.freeze_attn_v_mlp:
            for block in self.blocks:
                for p in block.attn.v_mlp.parameters():
                    p.requires_grad = False
        # Freeze the Q and K MLPs
        if cfg.attn_qk_mlp and cfg.freeze_attn_qk_mlp:
            for block in self.blocks:
                for p in block.attn.q_mlp.parameters():
                    p.requires_grad = False
                for p in block.attn.k_mlp.parameters():
                    p.requires_grad = False
        # FrozenMLP weights are reinitialized after apply() so they aren't overwritten
        # by _init_weights' std=0.02 normal. Use default nn.Linear init (Kaiming uniform).
        if self.frozen_mlp is not None:
            for lin in self.frozen_mlp.linears:
                lin.reset_parameters()
            for p in self.frozen_mlp.parameters():
                p.requires_grad = False

    @staticmethod
    def _sinusoidal(L, D):
        pos = torch.arange(L).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe = torch.zeros(L, D)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    @staticmethod
    def _init_weights(m, init_std: float):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=init_std)

    def forward(self, edges, return_hidden: bool = False, erase: dict | None = None):
        """edges: (B, seq_len) long ints in [0, vocab_size).

        Returns logits (B, n_classes), and optionally a list of residual-stream
        tensors at each depth: [embedding, after_block_1, ..., after_block_N],
        each (B, seq_len, d_model).

        If `erase` is provided with keys 'labels' (B,), 'targets' (K,), and
        'ridge' (float), the batch DoM directions for those K classes are
        projected out of h at every residual depth (post-embedding and after
        every block), before it enters the next block. Fully differentiable.
        """
        h = self.tok_emb(edges) + self.pos_emb
        h = self.drop(h)
        if erase is not None:
            h = erase_dom_projection(h, erase["labels"], erase["targets"],
                                     erase.get("ridge", 1e-4))
        hiddens = [h] if return_hidden else None
        for block in self.blocks:
            h = block(h)
            if erase is not None:
                h = erase_dom_projection(h, erase["labels"], erase["targets"],
                                         erase.get("ridge", 1e-4))
            if return_hidden:
                hiddens.append(h)
        h_final = self.ln_f(h)
        last = h_final[:, -1]
        if self.frozen_mlp is not None:
            last = self.frozen_mlp(last)
        logits = self.head(last)
        if return_hidden:
            return logits, hiddens
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
