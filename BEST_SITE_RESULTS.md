# Steering at the best token position

The requested protocol scans every residual depth and token position, then shows
steering at the best location. There is no last-token restriction and each edit
changes exactly one token. Locations are selected separately for each model seed,
latent, and checkpoint; a common location is not imposed across models.

![Best single-token steering](runs/emergence_best_site_summary/best_site_steering.png)

All 303 checkpoint scans are complete. Final held-out Brier skills, averaged
over all eight latents within each seed:

| Seed | Target−rest, strength 1 | Target−rest, selected strength | Target−reference, strength 1 | Target−reference, selected strength |
|---|---:|---:|---:|---:|
| 46 | -0.80385 | 0.26712 | -0.34412 | 0.35423 |
| 47 | -0.41909 | 0.90608 | 0.87385 | 0.92279 |
| 48 | -0.79377 | 0.51643 | 0.93547 | 0.93547 |

**Block 1, token 3** is selected for every latent of seed 47 when strength is
selected, for both contrasts. **Block 1, token 2** is selected for every latent
of seed 48 in those conditions. Seed 46 has no comparably reliable single cell:
the selected target−reference edits use tokens 3–4, mostly at block 1, with one
latent selecting block 2. Its target−rest edits mostly use block 1, tokens 2 or
4, with one unsuccessful case selecting block 5. Token indices are zero-based.

For target−reference means at strength 1, all three seeds select block 1; seed
46's selected token varies across latents among 2, 3, and 4. The other seeds use
tokens 3 and 2 respectively. Thus freeing the location recovers strong
single-token steering in two seeds, but does not produce equally strong behavior
in all three. Unit-strength target−rest steering remains poor even at its best
calibration-selected locations. Strength and contrast remain consequential.

The selected location maximizes raw target Brier skill on the final 256 original
training calibration pairs per latent. Its plotted score is then measured on the
original 462 held-out pairs per latent. No test score enters location selection.
Calibration pairs also contribute to the full training-support means, so
calibration scores themselves are not independent generalization measurements.

## What is scanned

- Depths 0–6: embedding state plus all six post-block residual states.
- Token indices 0–5, each edited separately: **42 locations**.
- Two analytic differences of class means: target minus the uniformly weighted
  other classes, and target minus the particular reference class (node 24).
- Unit-strength results select only depth/token.
- The separate strength-selected control scans strengths 1, 2, 4, 8, 16 as well
  as depth/token: 210 candidates per contrast, model, latent, and checkpoint.

The mean vectors use the complete 800,000 distinct training trajectories, with
FP32 forward passes and FP64 accumulation. No vector optimizer, learned direction,
or supervised fitting of vector coordinates is used. The finite strength search
is explicit and separate from the unit-strength result. Each selected direction
is a single 128-dimensional vector. No four-token vector enters this experiment.

The two panels in the overview keep the contrast definitions separate. Faint
lines are individual seed averages across all eight latents, and bold lines are
the three-seed averages. Raw negative scores are retained, with no smoothing.
The frequency-colored paired plots use a three-checkpoint moving average with
raw dots, again retaining negative scores.

## Plots and selected positions

- [Per-model comparison](runs/emergence_best_site_summary/per_seed.png)
- [Target-minus-rest, strength 1](runs/emergence_best_site_summary/rest_unit_steering_gain.png)
- [Target-minus-rest, strength selected](runs/emergence_best_site_summary/rest_scaled_steering_gain.png)
- [Target-minus-reference, strength 1](runs/emergence_best_site_summary/reference_unit_steering_gain.png)
- [Target-minus-reference, strength selected](runs/emergence_best_site_summary/reference_scaled_steering_gain.png)
- [Selected token positions](runs/emergence_best_site_summary/selected_positions.png)
- [Selected depths](runs/emergence_best_site_summary/selected_depths.png)
- [Final locations and scores](runs/emergence_best_site_summary/final_choices.csv)
- [Every location choice](runs/emergence_best_site_summary/all_choices.csv)
- [Unsmoothened crossing times](runs/emergence_best_site_summary/timing.csv)

“Best” means the best calibration score among the stated candidates. It does
not imply a globally optimal intervention beyond this grid. Locations can change
through training, so these are curves of the best available measured location,
not trajectories of one permanently fixed location. All seeds and all latents
are retained, including unsuccessful cases. The score is the existing exact
counterfactual-endpoint Brier skill, not the older, looser endpoint-reachability
metric in `src/dom_probe.py`.

## Reproduction

```bash
condor_submit_bid 2000 submit_emergence_best_site.sub
python3 plot_emergence_best_site.py runs/emergence_best_site_s*/history.json \
  --out-dir runs/emergence_best_site_summary
python3 -m unittest discover -s tests -v
```

Code commit `0676576`. The 101 saved checkpoints of seeds 46–48 are replayed
without retraining. Source models are in `/fast/fdraye/toy_model_linear/emergence_wide_s46`
(likewise s47/s48). Outputs are in `/fast/fdraye/toy_model_linear/emergence_best_site_s46`
(likewise s47/s48). The wrapper launches `remeasure_emergence_best_site.py` with
the corresponding source run and fresh output directory.

Condor jobs **17553159.0–2** completed with exit code 0 on **g197, i108, and i107**,
all NVIDIA H100 GPUs, in 235, 249, and 248 seconds respectively.
Calibration cubes, selected directions, exact pair IDs, selection metadata,
and histories are retained. Full class means at every depth/token are saved on
the cluster. Local `runs/emergence_best_site_s*` copies contain the review data;
the larger full class-mean archives remain on the cluster.

All sixteen local tests pass. New checks cover the shared forward pass for means
at all depths, calibration selection without clipping negative scores, separate
choices per latent, and agreement between explicit continuation and hooked
steering at embeddings and post-block states. Generalization is independently
checked during every checkpoint evaluation and the original curve is retained.

All **303 calibration/direction snapshots** were checked locally: every saved
selection exactly equals the argmax of its saved raw calibration cube; every
selected vector has shape `[8, 1, 128]` and is finite. All four selection modes
were checked at every checkpoint. Every generalization score is exactly unchanged
and all independent forward-pass discrepancies are zero. The final calibration
scores at depths 1–6 also match the earlier independent FP32 audit **exactly** for
every shared contrast, position, strength, and latent.
