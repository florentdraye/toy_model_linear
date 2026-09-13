# Stabilizing difference-of-means steering

The late dips in the width-512, seed-46 run disappear when steering is measured
on a causal exponential moving average (EMA) of model weights. Directions remain
ordinary target-minus-reference means at strength 1. This changes the evaluated
model; it does not smooth the measured scores or retrospectively repair the
original training trajectory.

## Protocol

At saved checkpoint t (every 200 training steps), update each floating model
parameter as `Wbar[t] = decay * Wbar[t-1] + (1-decay) * W[t]`, initialized from
step 0. Every evaluation consumes the entire earlier snapshot prefix, even when
measurements are split across jobs. No future snapshot contributes. Recompute
all class means from the averaged model using the same 800,000 training paths
(FP32 forward, FP64 sums). Select the intervention location among the same 56
candidates on training calibration pairs. Evaluate the same 462 held-out matched
pairs for each of 32 latents. Both mean estimation and location selection exclude
the test paths. Exact-patch and norm-matched random controls are retained.

The original trained weights, directions, and histories are preserved. No new
training was performed. All model inference ran on cluster H100s. Local work
was limited to analysis, plotting, and lightweight checks.

The comparison plots show unbounded raw Brier skill at every checkpoint, with
no curve smoothing, no monotonicity constraint, and a shared training-step axis.
The averaging coefficient was explored after inspecting the original noise:
0.5 and 0.8 on seven diagnostic checkpoints, followed by full trajectories for
0.8 and 0.95. These are exploratory repairs on the existing fixed evaluation
bank, not a preregistered comparison on an untouched test set.

## Evidence for the source of the dips

For latent 22, steps 12,200 to 12,400 at the same block-1 suffix location:
steering falls from 0.993517 to 0.509522; generalization and exact patching rise
from about 0.971 to 0.99996. The training-calibration steering score falls too.
The vector norm rises from 742.21 to 867.87, cosine similarity is 0.956576, and
the relative vector change is 0.360851. These are changes between model
checkpoints, not a consequence of drawing new evaluation examples.

The original training uses constant LR 0.001 after 100 warmup steps. Parameter
averaging demonstrates that averaging nearby models stabilizes this seed's mean
steering. It does not by itself establish which optimizer setting caused the
fluctuations; no learning-rate ablation has been run.

## Completed 0.8 trajectory

All 101 checkpoints (steps 0–20,000) are complete. At steps 10,000–20,000,
mean absolute adjacent-checkpoint change across the 32 steering curves falls
from 0.0358954 to 2.06e-10. Mean steering over this interval improves from
0.977778 to 0.9999999996. Final mean steering is 0.9999999998.
Generalization also stabilizes near 1. Some steering-transition dips remain
before step 10,000; the full raw plot shows them.

Local artifacts:
- `runs/emergence_weight_ema_0.8_s46/history.json`
- `runs/emergence_weight_ema_0.8_s46/figures/weight_ema_comparison.png`
- `runs/emergence_weight_ema_0.8_s46/step020000.pt`

## Longer averaging window

All 101 checkpoints of the 0.95 trajectory are now complete. The transition
and plateau are smooth, and final steering averages 0.99999999993; the worst
of the 32 final latent scores is 0.99999999981. The largest single-checkpoint
steering drop after step 4,000 is 0.00948, compared with 1.16471 originally.

| Evaluated model | Mean total downward movement after step 4,000 | Median steering midpoint | Median generalization midpoint |
|---|---:|---:|---:|
| Original | 2.83783 | 5,800 | 5,600 |
| Weight EMA 0.8 | 0.14643 | 6,600 | 6,400 |
| Weight EMA 0.95 | 0.00545 | 9,800 | 9,800 |

Downward movement is the sum of negative adjacent score changes, averaged over
32 latents. It is a descriptive stability measure, not a significance test.
Midpoints require raw score >=0.5 for three consecutive checkpoints. The longer
average reduces downward movement by approximately 99.8%, while delaying the
median steering midpoint by 4,000 steps. Its smooth curves describe the
averaged-model sequence, not the emergence time of the original raw models.

The final 0.95 training-selected locations are block 1 for 30 latents (19 token-2,
5 suffix, 6 all-token edits) and block 2 suffix for 2 latents. Location selection
near score 1 can have tiny margins, so these argmax labels need not imply a
unique scientifically preferred location.

Local artifacts:
- `runs/emergence_weight_ema_summary/weight_ema_comparison.png` and `.pdf`:
  original, 0.8 and 0.95 on the same axes; all 101 checkpoints and 32 latents.
- `runs/emergence_weight_ema_summary/validation.json`: numerical checks and timing.
- `runs/emergence_weight_ema_0.95_s46/figures/best_location_steering.png`:
  fixed site, selected site, and generalization, using display window 1.
- `runs/emergence_weight_ema_0.95_s46/history.json`, `step020000.pt`, `pair_ids.pt`.

Recreate the comparison locally:

```bash
python3 plot_emergence_stability.py \
  runs/emergence_large_locations_s46/history.json \
  runs/emergence_weight_ema_0.8_s46/history.json \
  --other runs/emergence_weight_ema_0.95_s46/history.json \
  --out-dir runs/emergence_weight_ema_summary
```

## Other seeds

The same 0.8 coefficient was checked at steps 19,600, 19,800 and 20,000 for
seeds 47 and 48. Final steering is 0.371681 and -0.365739, respectively, despite
generalization being approximately 1. Averaging removes prediction jitter but
does not guarantee a useful mean-edit representation across seeds.

## Validation and execution

- 19 lightweight tests pass, including parameter averaging and existing
  intervention/split/mean-estimation contracts.
- Averaging disabled reproduces the original 12,200 and 12,400 calibration
  matrices, location choices, and held-out scores exactly.
- Both full trajectories have 101 checkpoints with the original bank hash;
  all 6,464 location choices match their training-calibration argmax. All 64
  saved final vectors exactly equal their recomputed target-minus-reference
  means. Step-zero scores and calibration matrices equal the originals.
- All seven 0.8 pilot checkpoint scores/calibration matrices exactly match
  the corresponding full-trajectory chunks, verifying prefix consistency.
- The merger requires every chunk's full causal snapshot prefix, disjoint
  checkpoint ranges, consistent banks/configuration, and complete model and
  vector artifacts. Missing causal prefixes are rejected.
- Pilot: Condor 17553636.0 (0.5, g192) and .1 (0.8, g196).
- Full 0.8: 17553647.0–2 on g192/g196/g203; identity replay .3 on g203;
  seed confirmations .4 on g199 and .5 on i208.
- Full 0.95: 17553656.0–2 on g198/i202/g206.
- All compute devices: NVIDIA H100, reported 95,830 MiB.

Commands are recorded in `submit_emergence_weight_ema_pilot.sub`,
`submit_emergence_weight_ema_full.sub`, and `submit_emergence_weight_ema_long.sub`.
They launch `run_emergence_locations.sh`, which runs
`scan_emergence_locations.py --weight-ema-decay ...` with the listed source,
output directory and checkpoint range. Outputs live under
`/fast/fdraye/toy_model_linear/emergence_weight_ema_*`; originals remain under
`emergence_large_means_s46` and `emergence_large_locations_s46`.
