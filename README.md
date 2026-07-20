# toy_model_linear

Toy model for studying **linear representations of latent variables** in transformers.

The setup: a random layered DAG (default 5 layers, 1 source + 4×100 intermediate nodes, 10 outgoing edges per node) generates sequences. Input tokens are local edge indices along a path; the target is the final node id (100-way classification). Intermediate nodes at layers 1–3 are the latent variables to be linearly probed in the residual stream. Train/test splits are on **disjoint paths** to keep anti-memorization honest.

Code is hand-written and framework-light (plain PyTorch, no Hydra / Lightning / wandb). Every file is meant to be read and edited.

## Layout

```
train.py                 # CLI entry: python train.py --out-dir runs/exp1 ...
src/
  config.py              # GraphConfig, ModelConfig, TrainConfig, ProbeConfig dataclasses
  graph.py               # random layered DAG + path sampling
  data.py                # tokenized paths, disjoint train/test split
  model.py               # pre-LN causal transformer (attn variants, bilinear MLP/head, frozen MLP...)
  train.py               # training loop, AdamW / KL-Shampoo, LR schedule, DoM probe hooks
  probe.py               # linear probe cube: (depth × position × graph_layer)
  dom_probe.py           # difference-of-means probe tracker + steering
  adv_probe.py           # adversarial linear probes with gradient reversal
  ridge_adv.py           # closed-form ridge adversary
  leace.py               # LEACE concept-erasure baseline
  kl_shampoo.py          # KL-Shampoo optimizer
run_*.sh                 # experiment recipes (bash wrappers around train.py)
submit_*.sub             # HTCondor submit files
interactive_gpu.sub      # HTCondor interactive H100 session
analyze_*.py, plot_*.py  # offline analysis / plotting of runs/<name>/
probe*.py, apply_leace.py, continue_train.py  # standalone offline tools
```

Not tracked: `runs/`, `logs/`, `__pycache__/`, `.claude/`, and any `*.pt` / `*.png` / `*.pdf` outputs.

## Quickstart

```bash
pip install -r requirements.txt   # torch>=2.0
python train.py --out-dir runs/exp1
```

Every flag in `train.py`'s argparse maps 1:1 to a field in `src/config.py` — read that file for the source of truth.

Common flags:

- `--d-model 128 --n-heads 4 --d-ff 512 --n-blocks 4` — model shape
- `--weight-decay 0.1` — main anti-memorization lever (applied to 2D weights only)
- `--use-mlp / --no-use-mlp`, `--use-residual / --no-use-residual`
- `--mlp-activation {gelu,relu,relu2,tanh}`, `--mlp-type {standard,bilinear}`
- `--dom-probe-every-steps N` — track difference-of-means probe AUC through training
- `--adv-probe-lambda`, `--ridge-adv-lambda`, `--ortho-lambda` — adversarial / regularization terms

Outputs land in `--out-dir` (default `runs/<timestamp>-<tag>/`): `ckpt.pt`, `graph.pt`, `log.txt`, optional `dom_probe.pt`, and plots from the analysis scripts.

## Experiment recipes

The `run_*.sh` scripts are canned invocations for specific ablations (relu², no-residual, no-MLP, bilinear head, frozen MLP, LEACE erasure, etc.). They're the fastest way to reproduce a named experiment; open one and read it before running.

## HTCondor

Interactive H100 session:

```bash
condor_submit -i interactive_gpu.sub
```

Batch: `condor_submit submit.sub` (see the various `submit_*.sub` for parametrized ablations).

## Workflow (local edit ↔ cluster run)

1. Edit locally in VS Code.
2. Commit + push from local (SSH remote works from a laptop).
3. `ssh` into the cluster and `git pull` in this directory.
4. Launch runs via `run_*.sh` or `condor_submit`.

The cluster's outbound SSH to GitHub is blocked, so pulls from the cluster must use HTTPS (or an already-authenticated credential helper). Pushes from the cluster are avoided.
