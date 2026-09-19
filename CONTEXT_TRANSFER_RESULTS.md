# Cross-context loss-transfer measurements

## Corrected locally reachable context test

The original broad source pairs below allowed the graph-layer-2 parent to
change when graph-layer-3 latent `92` was inserted. A denser corrected test
removes that confound. Every probe-batch point first reaches one of the eight
reachable layer-2 parents with an edge to latent `92`. Its absent/present pair
shares the entire prefix through that parent, changes only the next local edge
so the graph-layer-3 node becomes `92`, and shares all later edge choices.
Each 128-point background contains exactly 16 points from every eligible
parent. The pool has 7,182 distinct training-supported pairs.

Across ten model seeds and every saved 100-step checkpoint, relative context
variation falls from 0.646 at initialization to a minimum of **0.317 at step
2,500**, precisely during acquisition (mean generalization gain 0.301). The
mean extra benefit is then strongly positive (49.0; RMS magnitude 85.6).
Variation rises again after acquisition; at step 10,000 it is 0.582, while the
RMS transfer signal has collapsed to 0.038 and generalization is 0.996. Thus
the clean result is a sharp context-independence transition during emergence,
followed by a late normalized ratio computed on a nearly exhausted loss signal.

The unsmoothed figure and compact data are under
`runs/context_variation_localparent_dense_summary/`. Recreate them with:

```bash
python3 plot_context_variation_dense.py \
  runs/context_variation_localparent_dense_s{46,47,48,49,50,51,52,53,54,55} \
  --out-dir runs/context_variation_localparent_dense_summary \
  --histories runs/emergence_optimized16_summary/seed_{46,47,48,49,50,51,52,53,54,55}/history.json
```

The remaining sections document the earlier, less controlled broad-context
experiment and should not be used for the locally reachable claim.

## Natural-frequency count scatter at gain 0.5

Latent `78` is the tracked target closest to ordinary `1/100` frequency. Its
training probability is 1.053%, giving an expected count of 21.6 in the real
2,048-example training batch. For each seed, the experiment selects the
checkpoint nearest generalization gain 0.5, fixes one 2,048-example batch whose
points all arrive at one of the five reachable parents of latent `78`, and
varies which examples take the next edge into `78`.

The plotted scalar is the unnormalized loss contraction
`q_i = g_i^T sum_j K_ij g_j`. Counts run from 0 through 64 in increments of 4,
with 128 independently sampled subsets at every nonzero count. The same paths
and subsets are used across ten model seeds, and four fixed held-out evaluation
points are averaged only after computing their scalar contractions.

The subset-mean extra transfer is nearly linear in count (`R^2 = 0.980`). Near
the natural count, at `m=20`, its mean is 1,535.7 and its subset SD is 1,661.2.
The corresponding `Var(q)/E(q^2)` is 0.539. Thus count strongly predicts the
mean transfer, but at this acquisition point the identity of the examples in
the subset remains comparably important.

The figure and compact tables are under
`runs/count_subset_scatter_l78_summary/`; raw per-seed tables are under
`runs/count_subset_scatter_l78_s46` through `s55`.

This experiment measures whether learning from examples containing a selected
graph-layer-3 latent helps held-out examples containing the same latent,
independently of the surrounding path context. It uses the raw width-128,
six-block optimized16 models for seeds 46–55 and all 16 established latents.

## Boundary and quantities

The encoder boundary is the complete residual state after transformer block 1,
with shape `[batch, 6, 128]`. The virtual encoder parameters are the token and
position embeddings plus block 1. Blocks 2–6 and the endpoint head are frozen.
Thus all paths from the moved parameters to the endpoint loss pass through the
measured boundary.

For held-out example `i` and source example `j`, the code computes

`T_ij = J_i J_j^T g_j`

with exact reverse- and forward-mode differentiation, where `g_j` is the
gradient of the 100-class endpoint cross-entropy loss at the boundary. Dense
kernels are never materialized. A unit test checks the derivative against a
small virtual encoder update.

Every source slot has a training-support path without the selected latent and a
matched path containing it. The pair has identical downstream edge choices.
Changing the exact count switches a subset of these paired slots. Batch size is
128; counts are 0, 16, 32, 64, 96, and 128. There are three background templates,
eight count replicates, four fixed held-out evaluation examples, and acquisition
panels nearest generalization gains 0.1, 0.5, and 0.9.

The through-training context test fixes `m=64`, uses 32 slot subsets per
background, and measures

`V = Var_S(q(S)) / E_S[q(S)^2]`, with
`q(S) = g_i^T [T_i(B_S) - T_i(B_0)]`.

Individual-contribution agreement uses disjoint eight-example reference and
measurement banks. Its control examples share no internal graph node with the
held-out evaluation path. The same no-shared-latent rule supplies the useful
alignment control.

## Results

The exact latent count has a reproducible positive average effect. From count 0
to 128, the mean actual held-out loss decrease changes by 0.00726 before,
0.01544 during, and 0.01347 after acquisition at virtual step size 0.0002. The
slope is positive for 81.9%, 86.9%, and 83.8% of latent/seed combinations.

The stronger context-independence prediction is not supported by these
measurements. Stage-aligned mean relative variation is 0.510 before, 0.464
during, and 0.501 after acquisition. Individual contribution agreement is also
small: its mean cosine is 0.020, 0.048, and 0.035, compared with no-shared-latent
controls of 0.004, 0.006, and 0.016. Useful alignment is positive but small
(0.012, 0.017, and 0.015), while its control stays within 0.001 of zero.

The interpretation is narrow: shared-latent count predicts average transfer,
but the identity of the contexts still explains substantial variation in the
encoder-boundary transfer. A positive count slope alone follows from additive
batch gradients and does not establish count sufficiency.

Actual-update linearity depends on the model seed. Across all raw measurements,
actual versus first-order transfer at step size 0.0002 has correlation 0.943;
some seeds remain nonlinear enough to give a 0.369 relative RMSE. The first-order
quantity itself is an exact derivative. Use the predicted panels for the local
claim and treat the actual panels as finite-step behavior. Seed 46 is well in
the linear regime at this step (correlation 0.9975, relative RMSE 0.079).

## Artifacts and reproduction

Raw local outputs are under `runs/context_transfer_v2_s46` through
`runs/context_transfer_v2_s55`. The compact ten-seed tables and figures are in
`runs/context_transfer_multiseed_summary`:

- `count_transfer_multiseed.{png,pdf}`
- `context_variation_multiseed.{png,pdf}`
- `transfer_geometry_multiseed.{png,pdf}`
- four `*_seed_summary.csv` tables containing within-seed estimates

Redraw the compact results without loading a model:

```bash
python3 summarize_context_transfer.py runs/context_transfer_v2_s{46,47,48,49,50,51,52,53,54,55} \
  --out-dir runs/context_transfer_multiseed_summary
```

The audit found 160 complete latent/seed directories and 3,680,436 raw rows.
Pilot jobs were clusters 17579473 and 17579486. The nine-seed sweep was cluster
17579489, with the single corrected seed-47/latent-92 repair in cluster
17579497. Jobs ran on H100 nodes g187, g188, g192, g196–g199, g201–g203,
g205–g206, i103, i104, i108, i202, i206, and i208. The malformed process
17579489.0 failed before computation and was replaced by the successful repair.
