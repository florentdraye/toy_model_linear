"""Entry point: train the toy model.

Run from project root:
    python train.py --out-dir runs/exp1
    python train.py --d-model 64 --n-blocks 2 --weight-decay 0.5 --out-dir runs/exp2

All flags map 1:1 to fields in src/config.py.
"""
import argparse

from src.config import GraphConfig, ModelConfig, TrainConfig
from src.train import train


def main():
    p = argparse.ArgumentParser()
    # ---- graph ----
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--nodes-per-layer", type=int, nargs="+",
                   default=[1, 100, 100, 100, 100])
    p.add_argument("--edges-per-node", type=int, default=10)
    # ---- model ----
    p.add_argument("--d-model", type=int, default=100)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--pos-enc", type=str, default="learned",
                   choices=["learned", "sinusoidal"])
    p.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-mlp", action=argparse.BooleanOptionalAction, default=True,
                   help="If False, transformer blocks are attention-only (no MLP).")
    p.add_argument("--mlp-activation", type=str, default="gelu",
                   choices=["gelu", "relu", "relu2", "tanh"],
                   help="Nonlinearity in each block's MLP (ignored when --mlp-type bilinear).")
    p.add_argument("--init-std", type=float, default=0.02,
                   help="Stddev for normal init of embeddings, linears, and bilinear head factors.")
    p.add_argument("--mlp-type", type=str, default="standard",
                   choices=["standard", "bilinear"],
                   help="standard: Linear-act-Linear. bilinear: (Wl x) * (Wr x) -> Linear.")
    p.add_argument("--suo-init", action="store_true",
                   help="Semi-orthogonal init for block attn QKV/proj and MLP linears.")
    p.add_argument("--attn-v-mlp", action="store_true",
                   help="Replace attention V projection with a 2-layer MLP.")
    p.add_argument("--attn-v-mlp-hidden", type=int, default=-1,
                   help="V MLP hidden dim (-1 -> d_model).")
    p.add_argument("--freeze-attn-v-mlp", action="store_true",
                   help="Freeze the per-block V MLPs (random fixed nonlinear projection).")
    p.add_argument("--attn-qk-mlp", action="store_true",
                   help="Replace attention Q and K projections with 2-layer MLPs each.")
    p.add_argument("--attn-qk-mlp-hidden", type=int, default=-1,
                   help="Q/K MLP hidden dim (-1 -> d_model).")
    p.add_argument("--freeze-attn-qk-mlp", action="store_true",
                   help="Freeze the per-block Q and K MLPs.")
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "sgd", "klshampoo"])
    p.add_argument("--frozen-mlp-layers", type=int, default=0,
                   help="0 disables; positive adds a frozen MLP between transformer and head.")
    p.add_argument("--frozen-mlp-hidden", type=int, default=-1,
                   help="hidden dim for the frozen MLP; -1 -> d_model.")
    p.add_argument("--frozen-mlp-activation", type=str, default="relu2",
                   choices=["relu2", "relu", "gelu", "tanh"])
    p.add_argument("--head-type", type=str, default="linear",
                   choices=["linear", "bilinear"],
                   help="Classification head: linear (h.w_c) or bilinear (h.A_c.h).")
    p.add_argument("--head-rank", type=int, default=8,
                   help="Low-rank factor for the bilinear head.")
    p.add_argument("--identity-unembed", action="store_true",
                   help="Replace ln_f and head with nn.Identity (logits = h[:, -1]). "
                        "Requires d_model == n_classes.")
    p.add_argument("--frozen-suo-unembed", action="store_true",
                   help="Initialize the linear unembedding semi-orthogonally and freeze "
                        "its weight and bias. For n_classes > d_model, W.T @ W = I.")
    # ---- train ----
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--lr-schedule-epochs", type=int, default=0,
                   help="If > 0, LR schedule uses this many epochs while training "
                        "still stops at --n-epochs. E.g. --n-epochs 20 "
                        "--lr-schedule-epochs 60 runs 20 epochs on the first "
                        "third of a 60-epoch cosine schedule.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--loss-type", type=str, default="ce", choices=["ce", "mse"],
                   help="ce: cross-entropy (default). mse: MSE(logits, one_hot(y)) "
                        "- no softmax anywhere.")
    p.add_argument("--balance-train-layer", type=int, default=-1,
                   help="If >=0, stratify each real training batch by this "
                        "graph layer's latent labels. Example: layer 4 with "
                        "100 latents and batch 8000 gives ~80 samples/latent.")
    p.add_argument("--adv-probe-lambda", type=float, default=0.0,
                   help="Adversarial-probe gradient-reversal coefficient (0 disables).")
    p.add_argument("--ortho-lambda", type=float, default=0.0,
                   help="Soft (semi-)orthogonality penalty on every nn.Linear weight "
                        "(sum ||W W^T - I||^2). 0 disables.")
    p.add_argument("--ridge-adv-lambda", type=float, default=0.0,
                   help="Closed-form ridge adversary: penalize linear decodability of "
                        "intermediate latents at every (depth, position). 0 disables.")
    p.add_argument("--ridge-adv-nu", type=float, default=1e-3,
                   help="Ridge regularizer for the closed-form probe (scaled by mean(diag(X^T X))).")
    p.add_argument("--ridge-adv-tau", type=float, default=0.05,
                   help="logsumexp temperature for pooling per-cell fit qualities; "
                        "smaller -> closer to attacking the worst cell.")
    p.add_argument("--dom-probe-every-steps", type=int, default=0,
                   help="Log DoM-probe AUC every K training steps for a fixed "
                        "sampled set of node latents (0 disables).")
    p.add_argument("--dom-probe-n-nodes", type=int, default=20,
                   help="Number of node latents sampled per intermediate graph layer.")
    p.add_argument("--dom-probe-seed", type=int, default=0,
                   help="Seed for sampling the fixed latent node set.")
    p.add_argument("--dom-probe-max-train", type=int, default=60000,
                   help="Cap probe train subset size (0 = all). Biggest lever "
                        "for per-probe forward-pass and finalize cost.")
    p.add_argument("--dom-probe-max-test", type=int, default=5000,
                   help="Cap probe test subset size (0 = all).")
    p.add_argument("--dom-steer-n-classes", type=int, default=0,
                   help="K target classes per layer for steering intervention; "
                        "0 disables (piggybacks on --dom-probe-every-steps).")
    p.add_argument("--dom-steer-alpha", type=float, default=1.0,
                   help="Multiplier on the DoM direction when steering.")
    p.add_argument("--dom-mlp-hidden", type=int, default=0,
                   help="Hidden dim for 1-layer MLP probes at every (d, p) cell; "
                        "0 disables.")
    p.add_argument("--dom-mlp-iters", type=int, default=100,
                   help="Adam iterations per MLP probe fit.")
    p.add_argument("--dom-mlp-lr", type=float, default=3e-2,
                   help="LR for MLP probe fitting.")
    p.add_argument("--dom-mlp-activation", type=str, default="relu",
                   choices=["relu", "gelu"], help="MLP probe activation.")
    p.add_argument("--dom-logreg-iters", type=int, default=0,
                   help="Adam iterations for multinomial LogReg probes at every "
                        "(d, p) cell; 0 disables.")
    p.add_argument("--dom-logreg-lr", type=float, default=3e-2,
                   help="LR for LogReg probe fitting.")
    p.add_argument("--dom-probe-grad-batch-size", type=int, default=200,
                   help="Batch size for class-balanced virtual-grad delta_h "
                        "probes (default 200).")
    p.add_argument("--dom-probe-grad-active-fracs", "--dom-probe-grad-active-frac",
                   type=float, nargs="+",
                   default=[0.01, 0.1, 0.5],
                   help="Positive fractions for probe-batch grad batches.")
    p.add_argument("--dom-probe-grad-lr", type=float, default=0.0,
                   help="Step size for the virtual SGD update; 0 -> use --lr.")
    p.add_argument("--ckpt-at-steps", type=int, nargs="+", default=[],
                   help="Save ckpt_step<N>.pt at each of these training steps.")
    p.add_argument("--erase-layer", type=int, default=-1,
                   help="Graph layer whose latents are erased via DoM projection "
                        "at every residual depth/position during training. -1 disables.")
    p.add_argument("--erase-n-features", type=int, default=0,
                   help="Number of node classes to sample from --erase-layer for erasure.")
    p.add_argument("--erase-seed", type=int, default=0,
                   help="Seed for sampling erased classes.")
    p.add_argument("--erase-ridge", type=float, default=1e-4,
                   help="Ridge added to Gram(W) diagonal to stabilize the projection solve.")
    p.add_argument("--out-dir", type=str, default="runs/default")
    args = p.parse_args()

    graph_cfg = GraphConfig(
        n_layers=args.n_layers,
        nodes_per_layer=tuple(args.nodes_per_layer),
        edges_per_node=args.edges_per_node,
        seed=args.graph_seed,
    )
    model_cfg = ModelConfig(
        vocab_size=args.edges_per_node,
        seq_len=args.n_layers - 1,
        n_classes=args.nodes_per_layer[-1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        pos_enc=args.pos_enc,
        causal=args.causal,
        use_residual=args.use_residual,
        use_mlp=args.use_mlp,
        mlp_activation=args.mlp_activation,
        init_std=args.init_std,
        mlp_type=args.mlp_type,
        suo_init=args.suo_init,
        attn_v_mlp=args.attn_v_mlp,
        attn_v_mlp_hidden=args.attn_v_mlp_hidden,
        freeze_attn_v_mlp=args.freeze_attn_v_mlp,
        attn_qk_mlp=args.attn_qk_mlp,
        attn_qk_mlp_hidden=args.attn_qk_mlp_hidden,
        freeze_attn_qk_mlp=args.freeze_attn_qk_mlp,
        frozen_mlp_layers=args.frozen_mlp_layers,
        frozen_mlp_hidden=args.frozen_mlp_hidden,
        frozen_mlp_activation=args.frozen_mlp_activation,
        head_type=args.head_type,
        head_rank=args.head_rank,
        identity_unembed=args.identity_unembed,
        frozen_suo_unembed=args.frozen_suo_unembed,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        lr_schedule_epochs=args.lr_schedule_epochs,
        train_frac=args.train_frac,
        seed=args.seed,
        device=args.device,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
        balance_train_layer=args.balance_train_layer,
        optimizer=args.optimizer,
        adv_probe_lambda=args.adv_probe_lambda,
        ortho_lambda=args.ortho_lambda,
        ridge_adv_lambda=args.ridge_adv_lambda,
        ridge_adv_nu=args.ridge_adv_nu,
        ridge_adv_tau=args.ridge_adv_tau,
        dom_probe_every_steps=args.dom_probe_every_steps,
        dom_probe_n_nodes=args.dom_probe_n_nodes,
        dom_probe_seed=args.dom_probe_seed,
        dom_probe_max_train=args.dom_probe_max_train,
        dom_probe_max_test=args.dom_probe_max_test,
        dom_steer_n_classes=args.dom_steer_n_classes,
        dom_steer_alpha=args.dom_steer_alpha,
        dom_mlp_hidden=args.dom_mlp_hidden,
        dom_mlp_iters=args.dom_mlp_iters,
        dom_mlp_lr=args.dom_mlp_lr,
        dom_mlp_activation=args.dom_mlp_activation,
        dom_logreg_iters=args.dom_logreg_iters,
        dom_logreg_lr=args.dom_logreg_lr,
        dom_probe_grad_batch_size=args.dom_probe_grad_batch_size,
        dom_probe_grad_active_fracs=tuple(args.dom_probe_grad_active_fracs),
        dom_probe_grad_lr=args.dom_probe_grad_lr,
        ckpt_at_steps=tuple(args.ckpt_at_steps),
        erase_layer=args.erase_layer,
        erase_n_features=args.erase_n_features,
        erase_seed=args.erase_seed,
        erase_ridge=args.erase_ridge,
    )
    train(graph_cfg, model_cfg, train_cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
