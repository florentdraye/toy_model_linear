# Difference of means, last token only

The [subsequent intervention audit](MEAN_INTERVENTION_RESULTS.md) recovers the
earlier successful mean control with target-minus-reference means at tokens 2–5.
It separates the two changes made in this last-token simplification.

Follow-up: the [completed depth scan](MEAN_DEPTH_RESULTS.md) finds that the same
mean edit fails at all six final-checkpoint depths on training calibration pairs.
Exact patches succeed only at block 6. The held-out replay below is preserved.

Completed on all 101 saved checkpoints of seeds 46, 47, and 48. Generalization
is **exactly unchanged** from the previous figure, including an independent
re-evaluation that matched every saved score exactly. The requested simpler
steering intervention does **not** work at transformer block 1: all 24 final
latent/seed steering scores are below zero before flooring. None crosses 0.5.
The exact last-token activation-patch control also fails, so this result cannot
be attributed solely to mean-vector estimation noise.

Open [steering_gain.png](runs/emergence_summary/steering_gain.png),
[per-latent controls](runs/emergence_summary/per_latent.png), or
[unclipped scores](runs/emergence_summary/raw_fidelity.png).
The previous optimized curves are preserved in
[emergence_optimized_summary](runs/emergence_optimized_summary/steering_gain.png).
There is no new emergence-time scatterplot because no steering threshold is
crossed; the old scatterplot remains only in the archived optimized directory.

## Exact definition

Use the entire 800,000-path training support, once per distinct trajectory.
At each model checkpoint and for each of 100 reachable graph-layer-3 latents:

```
mu_k = mean(last-token activation after block 1 | graph-layer-3 node = k)
v_k  = mu_k - (sum_{j != k} mu_j) / 99
h_last <- h_last + v_k
```

The positive set is uniform over trajectories through k. The negative distribution
is uniform over the other 99 latent labels and uniform over trajectories within
each label. Integrating it exactly by averaging class means uses all available
negative paths without bias from training frequencies or unequal path counts.
It is not the path-count-weighted mean of all negative trajectories.

For the eight selected latents, positive support sizes are 5,635; 5,595; 7,182;
13,630; 7,202; 18,435; 3,166; and 6,469 (latent order 53, 30, 92, 79, 7, 69, 48, 44).
Each negative support consists of the remaining training paths, between 781,565
and 796,834 distinct trajectories. The two groups need not have equal sizes
because their means are normalized separately. No repeated samples inflate counts.

The vector is **128-dimensional**, added only at token index **5**, with strength
1. Forward passes for means use FP32, sums use FP64. No endpoint supervision,
optimization, strength search, validation selection, or cross-checkpoint fitting
is used. The same support is reused at every checkpoint; test paths never enter
the means. No model is retrained.

Evaluation retains the original 462 held-out matched pairs per latent: reference
node 24 versus target k, with identical continuation choices. Steering adds the
target-versus-rest vector to the reference input; it does not explicitly remove
the reference representation. Generalization and steering use the same normalized
100-class Brier skill as before. Controls are recomputed at this last-token site.

## Outcome and limits

At step 10,000, averaged across all eight latents and three seeds:

| Measurement | Raw Brier skill |
|---|---:|
| Generalization | 0.98525 |
| Last-token difference-of-means steering | -0.87071 |
| Exact last-token activation patch | -0.87068 |

Every final steering score is negative and therefore appears at zero in the
bounded figure. The largest individual steering score anywhere in training is
only 0.0563 (rounded up). These are failures, not successful smooth emergence
curves. Raw negative scores are retained in JSON and `raw_fidelity.png`.

Even replacing this token's activation by the matched target's exact activation
is insufficient at block 1. That is consistent with later blocks continuing to
use the unchanged earlier tokens; this causal explanation has not been isolated
by additional ablations. It does not establish that last-token steering fails at
all transformer depths. No depth search was used to improve this result.

Plots retain the same three-checkpoint centered moving average, raw dots, and
unsmoothed seed-error bands as the previous figure. No monotonic fit, time shift,
or per-curve ceiling normalization is applied.

## Reproduction and checks

`submit_emergence_means.sub` submits `remeasure_emergence_means.py` against the
existing wide model snapshots and refined baseline histories. Jobs
**17552977.0–2** ran on **g199, g201, and i104**, all NVIDIA H100 GPUs. Each replay
took about 40 seconds for the measurements, plus setup and plotting. Outputs:
`/fast/fdraye/toy_model_linear/emergence_means_last_s46` (likewise s47/s48),
downloaded to the corresponding local `runs/` directories. Shared figures are
also in `runs/emergence_means_last_summary`; `runs/emergence_summary` is the new
current figure set. The old current directory was moved intact to
`runs/emergence_optimized_summary`; no old artifacts were deleted.

All **303 direction snapshots** were checked against their saved FP64 class
means: each is exactly the specified difference, has shape `[8, 1, 128]`, and is
finite. Class-mean snapshots have shape `[100, 1, 128]`. All training path IDs
are unique and disjoint from both sides of the evaluation pairs. Ten unit tests
pass, including unequal class-size balancing, position-specific means, frozen
weights, and a hook check proving only the last token changes. A tiny one-step
CPU smoke check verified the new training defaults and plotting end to end.

The new `train_emergence.py` defaults are `--direction-method uniform-mean`
and `--site last`. Historical submit recipes now explicitly select their old
optimized method and token layouts, preserving reproducibility. Code was
committed and synchronized to the cluster via a verified Git bundle and clean
fast-forward merge; GitHub push still fails SSH authentication.
