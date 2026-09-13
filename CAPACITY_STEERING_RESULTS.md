# More curves and larger models with analytic mean steering

The successful reference-mean result survives expansion from eight to **32
latents and three seeds**. Width 128 finishes with mean held-out Brier skill
**0.959979** at the original four-token block-1 intervention, or **0.960770**
with a training-selected location. All **909 checkpoint location scans** across
widths 128, 256, and 512 are complete. Selected-location steering averages
**0.960770 / 0.932784 / 0.301855** at each width's final checkpoint.

A near-perfect larger-model reproduction exists: **width 512, seed 46 scores
0.996236** across 32 latents, using only analytic means at strength 1. Its final
locations use block 1 for 20 latents and block 2 for 12. The other width-512
seeds score **0.226903 and −0.317575**, so increasing width did not make this
steering procedure reliable across seeds.

## Plots available locally

- [All capacities: raw selected steering and generalization](runs/emergence_capacity_summary/capacity_best_location.png)
- [Near-perfect width-512 seed 46, with 32 curves](runs/emergence_large_locations_s46/figures/best_location_steering.png)
- [Width 512: all three seeds](runs/emergence_large_locations_summary/best_location_steering.png)
- [Width 512: raw scores and exact-patch control](runs/emergence_large_locations_summary/raw_selected_steering.png)
- [Width 128: original intervention, 32 curves](runs/emergence_small32_means_summary/steering_gain.png)
- [Width 128: location selection](runs/emergence_small32_locations_summary/best_location_steering.png)
- [Width 256: location selection](runs/emergence_medium_locations_summary/best_location_steering.png)
- [Width 256: raw scores and patch control](runs/emergence_medium_locations_summary/raw_selected_steering.png)
- [Width 256: selected blocks and token sites](runs/emergence_medium_locations_summary/selected_locations.png)

Plots are under `runs/` in the local repository. Histories and plots are ignored
by Git; the full model snapshots and all-depth means are retained on the cluster.

## Fixed measurement protocol

All capacities use the same graph, split, reference node **24**, and 32 tracked
latent classes. Frequency ranks were declared before observing larger-model
outcomes and include the original eight ranks. The graph has seven layers,
100 nodes per nonsource layer, ten edges per node, and one million possible
six-token paths. Graph/data seeds are 0/314. Training samples latent classes
with the same geometric frequency ratio of ten and trajectories uniformly
within each class.

Every mean uses the **entire 800,000-path training split**, with FP32 forward
passes and FP64 accumulation. Class supports range from 1,596 to 18,435 paths;
the reference mean uses 2,409 paths. Each position has its own vector
`mu_target - mu_reference`. The main comparisons use strength **1**, without
vector optimization or strength fitting.

There are **462 distinct matched test pairs per latent**, fixed before training
and shared across checkpoints, seeds, and capacities. Both paths in every pair
belong to the held-out split. The fixed bank was expanded once for this study;
the old width-128 models were remeasured on that same expanded bank. Thus the
new eight-latent subset is not silently compared against different old examples.

- Evaluation-bank SHA256: `ad8f9e4c08dfca6b87c315ba4352973e6950b023f85a9690fdd11e596ce3c24e`
- Training-support SHA256: `1ed98eadf94fb37a9111a4f38b5b678fb22146a3fa7225bc0835fa2fff6ee46e`

The location search covers **56 conditions**: embeddings plus six transformer
blocks, crossed with six individual tokens, tokens 2–5, and all six tokens.
For each checkpoint, seed, and latent, the highest raw score on **256 fixed
training calibration pairs** selects the location. Those pairs are the final
quarter of the 1,024 training pairs and may overlap the mean support. Test
pairs are never used to estimate vectors, select locations, or select strength.
The entire evaluation bank remains fixed throughout training.

## Capacity comparison

All models have six blocks, four attention heads, ReLU-squared MLPs, batch
2,048, AdamW with learning rate 0.001, 100-step warmup, and weight decay 0.1.
Width 128 has **1,204,836** parameters; width 256 has **4,768,868**; width 512
has **18,974,820**. The old model snapshots end at 10,000 steps; the two new
widths were scheduled for 20,000, with evaluation every 200 steps. Compare the
shared 10,000-step checkpoint to assess capacity at equal training exposure.

Scores below are raw Brier skill averaged equally across all 32 latents and
three seeds. One is perfect and zero is uniform prediction; these are not
classification-accuracy percentages.

| Width | Step | Generalization | Fixed block 1, tokens 2–5 | Training-selected location |
|---|---:|---:|---:|---:|
| 128 | 10,000 | 0.985559 | 0.959979 | 0.960770 |
| 256 | 10,000 | 0.988122 | 0.779185 | 0.936761 |
| 256 | 20,000 | 0.995441 | 0.804738 | 0.932784 |
| 512 | 10,000 | 0.991654 | −0.034791 | 0.094454 |
| 512 | 20,000 | 0.994070 | 0.164511 | 0.301855 |

Width-256 final selected-site steering is **0.942008 / 0.858720 / 0.997623**
for seeds 46/47/48 respectively (see JSON for exact values). Of its 96 final
location choices, 95 use block 1; among those, 68 edit all tokens, 16 edit tokens 2–5, and
11 edit token 2 only. One choice uses block 2 with all tokens. The weakest
individual latent/seed score is 0.138600, so the pooled result is not a claim
of universal or perfect steering.

Final selected-location scores by model seed:

| Width | Seed 46 | Seed 47 | Seed 48 | Mean |
|---|---:|---:|---:|---:|
| 128 | 0.927571 | 0.988871 | 0.965868 | 0.960770 |
| 256 | 0.942008 | 0.858720 | 0.997623 | 0.932784 |
| 512 | 0.996236 | 0.226903 | −0.317575 | 0.301855 |

The width-512 seed-46 minimum over its 32 final latent scores is 0.936470.
Its 20 block-1 choices comprise nine all-token edits, nine suffix edits, and
two token-2 edits. Its 12 block-2 choices comprise five all-token edits, six
suffix edits, and one token-3 edit. There is no single universal winning token
position in this larger run.

At equal 10,000-step training exposure, increasing width improves mean
generalization slightly but lowers mean steering. Extending the larger models
to 20,000 steps does not reverse that ranking. These are results for the
stated graph, initialization, optimizer, and 56 intervention conditions.

For width 512, exact patches at the selected sites average **0.969679** while
mean steering averages **0.301855**. This leaves a large gap between replacing
a context-specific activation and adding a fixed class-mean shift, even after
location selection. The near-perfect seed-46 result shows that the simple
method can work extremely well; the full seed comparison shows its variability.

## Checks and controls

All **909 checkpoints** have the expected steps and finite calibration scores.
All **29,088 location choices** match the stored training-calibration argmax.
All **288 final selected vectors** were independently checked against the saved
class means and match exactly. The generalization replays match every source
history exactly, with maximum difference **0.0**. Downloaded pair banks match
the recorded hash, and test paths are disjoint from training/calibration.
Random-direction and exact activation-patch controls are retained, along with
all raw negative scores. All nine full location-scan jobs exited with code 0.

On the width-512 seed-48 checkpoint at step 5,400, the independent all-depth
mean computation exactly reproduces the saved training-time block-1 means
(maximum error **0.0**). All 32 selected vectors exactly equal the corresponding
target-minus-reference means. The location-selected unit edit scores −0.378718
versus generalization 0.942816. A separate training-only strength grid
`[0.5, 1, 2, 4, 8]` raises that early-checkpoint score to −0.135416; it does not
rescue the result and is not substituted for the main unit-strength curves.

The 17 existing scientific tests pass. A tiny CPU smoke check also confirms
that checkpoint resumption exactly preserves weights, sampler state, the fixed
evaluation bank, and final measurements relative to uninterrupted training.
All substantive model training and replay work runs on Condor H100 allocations.

## Execution record

Cluster output root: `/fast/fdraye/toy_model_linear`.

| Work | Condor jobs | Run directories |
|---|---|---|
| Width 512 training | `17553235.0–2` | `emergence_large_means_s46/s47/s48` |
| Width 128 expanded-bank replay | `17553247.0–2` | `emergence_small32_means_s46/s47/s48` |
| Early width-512 location audit | `17553270.0` | `emergence_large_locations_s48_step5400` |
| Width 256 training | `17553277.0–2` | `emergence_medium_means_s46/s47/s48` |
| Width 512 / 256 full location scans | `17553288.0–5` | `emergence_large_locations_s*`, `emergence_medium_locations_s*` |
| Width 128 full location scans | `17553304.0–2` | `emergence_small32_locations_s46/s47/s48` |
| Separate early strength control | `17553317.0` | `emergence_large_strength_s48_step5400` |

The launch recipes are the corresponding `submit_emergence_*means.sub` and
`submit_emergence_*locations.sub` files. Training runs execute
`run_emergence.sh`; fixed-location replays execute `run_emergence_means.sh`;
location scans execute `run_emergence_locations.sh`. All bids are 2,000.
`monitor_emergence_capacity.py` releases the prepared, held scans after each
parent training job exits successfully, without occupying a GPU while waiting.

Width-512 seeds 46/47 began on `i104` and resumed from steps 11,000/10,400 on
`g195`/`g203` due to slow throughput. Their JSON records the resumption. Seed
48 trained on `g206`. Width 256 trained on `g195`/`g202`/`g205`. All were H100s.
The completed large-model scans ran on `g195`/`g205`/`g205`; medium scans ran
on `g199`/`i208`/`g195`; small-model scans ran on `i208`/`i206`/`g197`.
An initial partial seed-48 scan on `i104` was preserved under
`archive/emergence_large_locations_s48_i104` before restarting that frozen-model
scan elsewhere. A similarly slow partial seed-47 scan on `i208` was preserved
under `archive/emergence_large_locations_s47_i208`; that scan restarted on
`g205`. No completed result was overwritten.

Code commits were synchronized to the cluster by verified Git-bundle
fast-forwards because GitHub SSH push was unavailable in this session.
