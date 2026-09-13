# Steering and generalization at graph layer 3

The completed 10× frequency experiment produces distinct, smooth learning curves
for eight latents across three seeds. Steering and generalization emerge on similar
timescales for most latents, but the result is **not exact simultaneity**. All 24
latent–seed pairs cross both 0.5 thresholds. The median paired ratio
`t_steer / t_gain` is **0.971**; 18/24 ratios lie within 20% of one. The most
frequent latent consistently steers later, while several rare latents steer earlier.
These are descriptive results on one graph, not a universal timing law.

Open the [side-by-side curves](runs/emergence_summary/steering_gain.png),
[per-latent overlays and controls](runs/emergence_summary/per_latent.png), or
[emergence-time comparison](runs/emergence_summary/emergence_times.png).
PDF versions are in the same directory. The full, unsmoothed paired measurements
are in [timing.csv](runs/emergence_summary/timing.csv).

Median crossing steps across seeds 46, 47, and 48:

| Graph-layer-3 latent | Training frequency | Generalization | Steering |
|---|---:|---:|---:|
| 53 | 2.489% | 1,695 | 2,501 |
| 30 | 1.798% | 1,798 | 2,162 |
| 92 | 1.298% | 2,543 | 2,516 |
| 79 | 0.937% | 3,615 | 3,503 |
| 7 | 0.677% | 3,514 | 3,406 |
| 69 | 0.489% | 5,792 | 5,136 |
| 48 | 0.353% | 3,826 | 3,653 |
| 44 | 0.255% | 6,861 | 5,665 |

The median ratio above is calculated within each latent–seed pair, not by dividing
the columns of this table. Frequency is not the only determinant: latent 48 learns
before latent 69 despite being rarer. No latents were removed to enforce ordering.

At step 10,000, mean generalization fidelity is **0.9853** and mean optimized
steering fidelity is **0.99996**. Across all 24 individual measurements, the minima
are 0.9206 and 0.99936, respectively.

## What is measured

The unchanged random DAG has seven node layers, 100 nodes in each non-source
layer, and ten outgoing edges per node. Graph layer **3** is the middle latent.
There are one million possible six-edge trajectories; an 80/20 split makes
training and evaluation paths disjoint. Training first draws a latent from a
shuffled geometric categorical law, then a uniform training path conditional on
that latent. This removes the confound from unequal numbers of paths reaching
different nodes. Across the selected latents and three seeds, realized training
frequencies are within 2.03 binomial standard errors of their requested values.

Every latent is compared with reference node **24**, using exactly the same suffix
edge choices. The direction is fitted independently at each of 101 checkpoints.
There are 1,024 calibration pairs per latent, with 768 used for supervised fitting
and 256 for validation selection. Both paths of each calibration pair are in the
model's training support. Both paths of each of the **462 distinct evaluation
pairs per latent** are outside training. The count is capped equally by finite
matched-pair support; pairs are not duplicated to inflate the sample size.

The model is a six-block causal transformer with width 128, four heads, ReLU²
MLPs of width 512, and residual connections. The intervention is after
**transformer block 1**, at token indices **2–5**. Each latent has one constant
**512-dimensional vector**, reshaped to four × 128 for insertion. It is shared
across contexts, but it is not a single-token vector and is fitted specifically
for swaps from reference node 24.

The initial vector is the exact least-squares mean matched hidden displacement.
Projected Adam then fits the frozen downstream network for 400 probability-loss
steps. A further 400 log-loss steps improve gradients when the prediction is
confidently wrong. Validation probability error chooses the best iterate,
including the original vector. The norm is bounded by the RMS size of a natural
matched hidden-state change. No test examples or future checkpoints select the
vector, its amplitude, or its iteration. This is **supervised controllability**,
not merely an unsupervised representation probe or a guaranteed global optimum.
The mean-vector control remains visible beside the optimized result.

Both headline scores are normalized multiclass Brier skill:

```
gain = 1 - mean ||p - one_hot(counterfactual endpoint)||² / (1 - 1/100)
```

For generalization, `p` is the prediction on the held-out target trajectory.
For steering, it is the prediction on the edited reference trajectory. Uniform
prediction scores zero, and perfect target prediction scores one. Raw negative
scores are preserved in JSON and `raw_fidelity.png`; the headline floors them at
zero. The paired-effect score from granularity is also retained in JSON, but is
not the headline: learning the common reference alone can contribute about half
of that score without learning the target.

The plots average within each latent across seeds, use a documented three-point
centered moving average, and retain raw checkpoint dots. Shading is ±1.96 seed
standard errors, descriptive with only three seeds. Timing uses **unsmoothed raw
scores**, an absolute 0.5 threshold sustained for three checkpoints, and linear
interpolation of the first crossing. There is no monotonic fitting, time alignment,
or normalization by each curve's own ceiling. Per-pair errors are not treated as
independent-seed confidence intervals because pairs can share individual paths.

## Controls and limits

The earlier block-3 pilot generalized but could not be steered effectively by a
constant mean shift. Exact activation patches worked, motivating a validation-only
scan of five depths, three token layouts, and two norm limits. Block 1 was effective.
The initial 128-dimensional single-token variant subsequently proved unstable in
one seed and left a rare tail unlearned under 30× skew and weight decay 1.0.
Those results remain in [the single-token comparison](runs/emergence_token_summary/steering_gain.png).

The final condition uses a 10× frequency range and weight decay 0.1. Its matched
uniform-frequency control uses the same graph, architecture, seed 46, and vector
fitting procedure. At step 10,000 its mean generalization/steering scores are
**0.838 / 0.463**; it is not fully solved. See
[the uniform control](runs/emergence_refined_uniform/figures/steering_gain.png).
Thus the experiment does not establish frequency-independent or universal
co-emergence. The two conditions can develop different internal computations;
the control should not be presented as a converged replication of the skewed result.

## Artifacts and reproduction

The final local directories are `runs/emergence_refined_s46`, `s47`, and `s48`.
Each contains `history.json`, `graph.pt`, `final_model.pt`, and all 101 fitted
direction tensors in `directions/stepXXXXXX.pt`, each shaped `[8, 4, 128]`.
All **303 snapshots / 2,424 latent vectors** were checked for shape, finite values,
and compliance with their norm budgets. Latent order is recorded in each JSON.
Full training snapshots, split indices, and pair IDs remain on the cluster under
`/fast/fdraye/toy_model_linear/emergence_wide_s46` (likewise s47/s48).

The recommended recipes are `submit_emergence_robust.sub` followed, after all four
training jobs finish, by `submit_emergence_refine.sub`. They reproduce the three
skewed seeds and matched uniform control; change output names for a fresh run.
Both submit files passed Condor dry-run validation. Plot with:

```bash
python3 plot_emergence.py runs/emergence_refined_s*/history.json \
  --out-dir runs/emergence_summary --smooth-window 3
```

Training jobs were **17552865.0–2** on H100 nodes g206, g202, and i202; refinements
were **17552889.0–2** on g206, g202, and g203. The matched control used **17552868**
on g190 and **17552892** on i204. The environment recorded PyTorch 2.13.0+cu130.
Commands and complete configurations are preserved in the submit recipes and
result JSON. All heavy computation ran on allocated compute nodes.

Six scientific-contract tests pass, covering sampling frequencies, path isolation,
finite support, teacher scoring, exact interventions, frozen model weights, and
both fitting objectives' validation/norm constraints. A tiny CPU smoke check also
verified that resuming training reproduces uninterrupted weights and sample counts
exactly. Code is committed locally and synchronized to the cluster. GitHub push
failed SSH authentication, so synchronization used verified Git bundles and clean
fast-forward merges; GitHub has not been updated.
