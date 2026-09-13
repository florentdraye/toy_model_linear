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

### Focused latent-frequency / steering experiment

`train_emergence.py` reuses the DAG and transformer, with one selected graph
layer (default 3), one fixed post-block intervention (default block 3 of 6),
and paired steering/generalization curves. It does not run the old probe cube.

Training draws a latent first from a shuffled geometric law (100× most-to-least
frequent), then a uniform path from that latent's training support. This corrects
for unequal numbers of paths reaching different graph nodes. The complete path
support is split before sampling; duplicates within training are intentional.
`--frequency-ratio 1` is the matched uniform-frequency control. Graph, frequency
assignment, train/test split, and measurement banks use `--data-seed` /
`--graph-seed`; changing `--seed` changes model initialization and training draws.

For each of eight frequency-ranked latents, compare a trajectory through it to
one through the same most-frequent reference latent, keeping all subsequent edge
choices identical. Both evaluation paths are held out. Direction fitting uses
separate pairs whose two paths are in training. At every checkpoint,
`v_k = mean(h_target - h_reference)` is the exact least-squares optimal constant
additive translation of those matched activations. This is the `mean` control.
The default `--direction-method optimized` initializes there and fits a single
constant vector through the frozen downstream network to the correct counterfactual
endpoint, using the first 75% of calibration pairs. The remaining 25% select the
best iterate per latent. Projected Adam constrains the norm to the RMS size of a
natural matched hidden change (no unbounded strength search). `--direction-steps`
and `--direction-lr` control the solve; validation improvements, selected iterations,
and norm budgets are saved. This is a numerical optimization, **not** a global
optimality guarantee. It measures supervised controllability, whereas the mean
control measures representation-derived steering. Model weights are frozen; no
final-test pairs or future checkpoints are used. Final test data are scored only
after fitting and validation selection.

Default `--site suffix` edits the residual positions from the latent's token to
the end, at one transformer depth. The vector is the flattened residual slice
(four positions × 128 dimensions by default), reshaped on insertion. This lets
the intervention reach parallel copies of the latent at later token positions.
`--site token` gives a single d_model-dimensional vector at token layer−1;
`--site all` includes every token. The site is fixed throughout each run.

Both scores use the normalized multiclass Brier skill
`1 - mean||p - onehot(target_endpoint)||² / (1 - 1/n_classes)`, where p is
either the prediction on the held-out target input (generalization) or the
edited reference input (steering). Uniform prediction scores 0, perfect target
prediction scores 1, and mistakes can score below 0. Headline plots
floor negative scores at 0; JSON and `raw_fidelity.png` retain them. Coincident
teacher endpoints are retained to penalize off-target effects. An exact,
context-dependent activation patch and a norm-matched random-vector control
diagnose intervention-site limits and nonspecific effects. The paired-effect
fidelity from granularity, `1 - Σ||Δp - Δonehot||² / Σ||Δonehot||²`, is retained
in `gain_effect_raw` / `steer_effect_raw`. It is not the headline here: learning
only the common reference can contribute about half of that score even when the
target latent is unlearned. Target fidelity removes that reference-only credit.
Target and reference accuracies are saved separately.

Submit on the cluster (never train on the login node or laptop):

```bash
mkdir -p /fast/fdraye/toy_model_linear/logs
condor_submit_bid 2000 submit_emergence.sub
```

The wrapper uses granularity's existing PyTorch venv. Results are written to
`/fast/fdraye/toy_model_linear/emergence_opt_pilot`: incremental `history.json`,
resumable `resume.pt`, exact pair IDs in `banks.pt`, one fitted vector per latent
per checkpoint in `directions/`, and PNG/PDF figures. Use `--resume` with the same
configuration to continue an interrupted run. `--save-models` additionally retains
model snapshots for offline measurements. Pair counts are capped equally across
latents when a selected latent has fewer distinct pairs than requested; actual
counts are recorded. The learning rate is constant after
100 warmup steps. Existing result directories are protected against accidental reuse.

```bash
python3 -m unittest discover -s tests -p test_emergence.py -v
python3 plot_emergence.py runs/emergence_seed*/history.json --out-dir runs/emergence_figures
```

Plots default to raw checkpoint means; `--smooth-window 3` optionally adds a
documented centered moving average with raw dots retained. Multiple seeds are
averaged within the same latent, with seed uncertainty bands. `timing.csv` uses
unsmoothed raw scores, an absolute 0.5 threshold sustained for three checkpoints,
and explicit censoring. A low ceiling is never rescaled into apparent emergence.
Timing agreement is a hypothesis to test, not a constraint imposed on the plots.

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
