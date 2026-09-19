# Graph traversal: existing steering and gain data

## Status

- Existing measurements exported to the `learning-across-contexts` repository at the user's request.
- Date: 2026-09-19.
- Task project: `toy_model_linear`, experiment revision `c663673`.

## What I loaded

- Source files: `runs/emergence_optimized16_summary/seed_{46..55}/history.json` and
  `runs/emergence_optimized16_summary/figures/timing.csv`.
- Relevant code: `train_emergence.py`, `refine_emergence.py`, `src/emergence.py`,
  `plot_emergence.py`, and the fixed recipe in `run_optimized16.sh`.
- Runs: model seeds 46–55; graph seed 0; split/data seed 314.
- Checkpoints: steps 0–10,000 inclusive, every 100 steps (101 per seed).
- Raw model weights; no weight EMA or score EMA.

## Measurement definitions

- Latent: the mutually exclusive graph-layer-3 node reached after input token 2.
  Sixteen predeclared frequency ranks 29–99 are measured; reference node 24 is
  the most frequent latent.
- Gain: normalized multiclass Brier skill of the unedited held-out target path
  against its endpoint label.
- Steering: the same skill after adding one context-independent optimized vector
  to the matched reference path.
- Normalization: `1 - mean(||p-one_hot(y)||^2)/(1-1/100)`. Uniform prediction is
  zero, perfect prediction one, and negative values are retained in the export.
- Steering boundary: residual stream after transformer block 1, token positions
  2–5, tensor shape `[4,128]`. The vector is initialized by the matched-pair mean
  displacement, fit for 400 projected-Adam Brier steps, then refined for 400
  cross-entropy steps. A fixed training validation quarter selects the iterate.
- Raw checkpoint training cross-entropy is available. Step 0 is the recorded
  zero placeholder; later values average the preceding 100 updates.

## Exported tables

- `steering_gain_curves.csv`: one row per seed/checkpoint/latent with raw curve
  metrics, controls, counts, norms, and fitting diagnostics.
- `emergence_times.csv`: one row per seed/latent. Emergence is the first raw 0.5
  crossing sustained for three checkpoints, linearly interpolated across the
  crossing interval. No smoothing or own-ceiling normalization enters timing.

## Missing or uncertain information

- The saved histories do not contain classification accuracy after steering;
  `accuracy` is the unedited target-path endpoint accuracy.
- Pair-level losses are summarized as Brier-skill means and standard errors;
  individual pair responses were not exported by this experiment.
- The established steering tensor is a four-token slice. For later Jacobian
  transfer identities, the complete post-block-1 residual `[6,128]` may be
  required so every downstream-to-loss path crosses the chosen boundary.

## Proposed later measurements

The existing graph, fixed held-out pair bank, checkpoints, latent definitions,
and post-block-1 boundary can be reused. Later transfer/count, relative context
variation, contribution agreement, and useful-alignment measurements require new
VJP/JVP computations from saved models. Their evaluation/source/reference banks,
batch counts, subset identities, dot products, norms, and losses should remain in
this task repository until separately approved for export.
