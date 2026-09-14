# Kernel response versus latent count in a training batch

This experiment fixes one held-out reference input per tracked latent, computes
its kernel responses to a large training pool, and varies the identities and
latent counts of the examples contributing to the response. It uses width 512,
seed 46, and the EMA-0.95 model sequence from the stability experiment.

## Quantities

Let `z_i` denote all 100 endpoint logits, `J_i = d z_i / d theta`, and
`g_i = d CE_i / d z_i = softmax(z_i) - one_hot(y_i)`. The empirical logit NTK
is `K_ij = J_i J_j^T`, using all model parameters with the Euclidean inner
product. At each frozen checkpoint we measure two scalar contractions:

1. `q_ij = [K_ij g_j]_{y_i}`: correct-endpoint component of the logit response.
2. `r_ij = g_i^T K_ij g_j`: reference cross-entropy loss contraction.

The scatter y-values are the unnormalized sums over a sampled batch. The
negative of these sums times `learning_rate / batch_size` gives the
first-order SGD logit/loss change. No optimizer update was applied. These are
not Adam responses and do not incorporate clipping or weight decay.

The main plot uses the correct-endpoint component because `K_ij g_j` is a
100-vector in this classification model. Companion plots contain the loss
contraction. The full 100-vector is saved for the independent 32-example
aggregate validation; full vectors for every pool example are not materialized.

Projected products use Jacobian-vector products and reference parameter
gradients, following the [JVP/VJP kernel-product identity in the PyTorch
NTK tutorial](https://docs.pytorch.org/tutorials/intermediate/neural_tangent_kernels.html).
Forward/JVP computation uses FP32 with math SDPA; softmax, contractions, and
batch summation use FP64. No finite-difference approximation or random kernel
projection is used.

## Fixed protocol

- All 32 previously tracked graph-layer-3 latents, with the same frequency order.
- One reference example per latent, selected from the original held-out target
  bank using seed 7141 before evaluating its predictions. It stays fixed across
  all stages and does not occur in the training pool.
- Three stages per latent: saved checkpoints nearest its Brier generalization
  gains 0.1, 0.5 and 0.9. Actual gains and steps appear on the plots. Stage
  selection uses the latent-average gain, not the reference example's gain.
- A common pool of 102,400 distinct training examples: 1,024 per each of the
  100 reachable latent classes, selected without replacement using seed 7139.
- Batch size 16,384, eight times the original training batch size.
- Target-latent counts: 0, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 12,288,
  and 16,384. There are 64 independent batches at each count.
- At each x-value the full vector of all 100 latent counts is fixed. Other
  classes share the remaining slots according to the original training
  probabilities, renormalized to exclude the target, with deterministic
  largest-remainder rounding.
- Within each stratum, batch examples are sampled with replacement, as in the
  original training sampler. Thus batches contain 16,384 entries, not
  necessarily 16,384 distinct paths. All stages use identical sampled batches.
- There is no smoothing, plot-value clipping, horizontal jitter, or outcome
  selection. The orange line is the exact conditional mean over the fixed pool.

The two pilot latents are the highest- and lowest-frequency tracked classes
(53 and 44). The remaining 30 follow exactly the same protocol. The overview
selects eight frequency-spaced latents; individual plots and PDF pages contain
all latents.

## Completed results

All 32 latents and all 96 reference/checkpoint combinations are complete:
9,830,400 per-point kernel responses and 61,440 batch/stage scatter points
per projection. The same 640 sampled batches per latent are reused at all
three stages. All 32 Condor jobs exited with code 0.

Count predicts the response strongly around gain 0.5. To compare the residual
spread with the count effect, divide the mean within-count sample SD by the
range of the pool-conditional means over the full 0–16,384 count sweep:

| Projection | Before (gain near 0.1) | Midpoint (near 0.5) | After (near 0.9) |
|---|---:|---:|---:|
| Correct-endpoint logit: median SD / count span | 2.57% | 0.98% | 1.57% |
| Reference loss contraction: median SD / count span | 2.53% | 0.96% | 1.63% |

For the logit projection, the after-stage ratio exceeds the midpoint ratio
in all 32 latents; it is smaller than the before-stage ratio in 17 of 32.
Count dependence is already visible at the before stage for many references.
These are relative spread measures over a wide count range, not probability
estimates or a proof of exact count sufficiency.

The median absolute logit spread after removing the batch-averaging factor
(`SD(sum)/sqrt(N)`) is 32.47, 60.83 and 34.84 across the three stages. The
corresponding loss-contraction values are 26.19, 22.69 and 1.82. The latter also
contains the reference loss gradient, which becomes smaller as that reference
is learned: its median CE loss is 2.603, 0.454 and 0.041. A decrease in the
loss contraction alone therefore cannot identify context independence.

Main artifacts:
- [All 32 logit scatter plots, PDF](runs/kernel_batch_counts_ema095_s46/figures/all_latents_logit.pdf).
- [All 32 loss-contraction scatter plots, PDF](runs/kernel_batch_counts_ema095_s46/figures/all_latents_loss.pdf).
- [Eight frequency-spaced examples](runs/kernel_batch_counts_ema095_s46/figures/overview_logit.png).
- [Raw scatter CSV](runs/kernel_batch_counts_ema095_s46/figures/scatter_points.csv).
- [Summary CSV](runs/kernel_batch_counts_ema095_s46/figures/summary.csv).
- [Audit JSON](runs/kernel_batch_counts_ema095_s46/audit.json).

## How to interpret the scatter

At a fixed reference and checkpoint, write `a_j` for either scalar response.
Conditional on the class-count vector `n`, the pool-conditional moments are:

```
E[sum_j a_j | n] = sum_l n_l * mean(a | latent=l)
Var[sum_j a_j | n] = sum_l n_l * Var(a | latent=l)
```

The second equality follows from independent sampling with replacement within
strata. These moments are saved alongside the sampled responses. Since other
class proportions vary deterministically with the target count, the expected
line is nearly affine by construction. A straight line alone does not show
that latent counts become sufficient through learning.

The relevant observations are the vertical spread at each fixed count, its
size relative to the count-dependent change, and how these change across the
three stages. The summary also includes `SD(sum)/sqrt(batch_size)`, equivalent
to `sqrt(batch_size)*SD(batch_mean)`, so the spread can be examined without the
usual averaging benefit of a larger batch.

Increasing target count at fixed total size necessarily removes other latent
examples. The slope therefore compares target and background contributions;
it is not a derivative holding every other count fixed. This study concerns
one model seed and one reference example per latent, conditional on the fixed
training pool. It does not establish a statement for every possible reference,
training example, or model seed.

## Files and reproduction

Cluster output root: `/fast/fdraye/toy_model_linear/kernel_batch_counts_ema095_s46`.
The corresponding downloaded output is `runs/kernel_batch_counts_ema095_s46`.
Each `latent_<id>/` contains:

- `history.json`: reference identity, source model, actual stages, CE loss and
  probabilities of the reference, seeds, full class probabilities, and checks.
- `pool.npz`: pool path IDs, latent classes, endpoints, and reference tokens.
- `kernel_before.npz`, `kernel_midpoint.npz`, `kernel_after.npz`: per-point
  products of shape [102400, 2], with columns logit then loss contraction.
- `scatter.npz`: responses [10 counts, 64 batches, 3 stages, 2 projections],
  exact class-count vectors, conditional means and predicted standard deviations.

Run `python3 plot_batch_kernel.py runs/kernel_batch_counts_ema095_s46
--expected-latents 32` on one line to recreate all figures and CSV exports.
The `figures/` folder contains individual three-panel PNGs, multipage PDFs,
frequency-spaced overviews, all raw scatter points, and summary statistics.

Submit recipes: `submit_batch_kernel_pilot.sub` and `submit_batch_kernel_full.sub`.
They launch `run_batch_kernel.sh` and `measure_batch_kernel.py` on cluster H100s.
Pilot Condor cluster: 17557153; full scan cluster: 17557162.

## Validation

- 21 lightweight tests pass. The kernel test explicitly constructs the full
  tiny-model Jacobians and compares both projected products to the implementation.
- A constant-within-latent synthetic control has exactly zero within-count
  variation; count preservation and deterministic sampling are checked.
- Every stage independently sums gradients over 32 training examples first,
  then differentiates the reference outputs in that direction. Both scalar
  contractions are compared to the sum of the corresponding per-point products.

The downloaded artifact audit checks all reference/train exclusions, all unique
pool IDs and their common hash, full batch-count vectors, stage selection,
and recomputed conditional moments. It replays all sampled batches for the
first latent from the saved seeds and products. All checks passed. Across 96
stages, the maximum relative error of the independent kernel contraction check
is 0.00013288 (0.0133%). The sampled conditional means have standardized Monte
Carlo error RMS 1.054; 94.48% lie within two predicted standard errors.

Recreate this audit with:

```bash
python3 audit_batch_kernel.py runs/kernel_batch_counts_ema095_s46 \
  --trajectory runs/emergence_weight_ema_0.95_s46/history.json \
  --banks runs/emergence_capacity_summary/banks.pt
```

Compute hosts were g191, g197, g202, g206, i103, i107, i108, i201 and i202;
all devices were NVIDIA H100. Per-latent execution metadata is retained in
its history file. No model training or model inference ran on the Mac.
