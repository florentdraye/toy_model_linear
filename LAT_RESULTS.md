# LAT steering and inter-latent geometry

## Status

Completed and audited a 21-checkpoint replay of the **original**, unaveraged
width-512 seed-46 model, at steps 0, 1000, ..., 20000. All 100 graph-layer-3
latent classes get LAT axes; the existing 32 targets retain their fixed 462
held-out intervention pairs. No training or model execution runs on the Mac.

All 21 checkpoints are complete and downloaded, along with an independent
hook-based replay, strength-1 evaluation, and balanced split-bank stability
audit. All 22 Condor jobs exited successfully. Open the
[steering curves](runs/emergence_lat_s46/figures/lat_steering.png) and
[mean absolute cosine curves](runs/emergence_lat_s46/figures/lat_geometry.png).

LAT preserves strong final steering with calibrated strength, but **does not
make the latent axes more orthogonal than the mean directions** in this run.

Local artifacts: `runs/emergence_lat_s46/`.
Cluster artifacts: `/fast/fdraye/toy_model_linear/emergence_lat_s46/`.

## Estimator and pairing

For each class k, form 792 fixed training-only pairs: eight contrasts against
each of the other 99 classes. Both paths are in the 800,000-path training split,
and the suffix edge tokens after graph layer 3 match exactly. Prefix pairs and
suffixes are sampled uniformly subject to both paths belonging to training.
Negative classes receive equal weight. There is no privileged reference class
in the axis estimator or its geometry.

The bank has shape [100, 792, 2], with off/on path IDs. There are 141,446 distinct
paths overall and 612–777 distinct positive paths per class; individual paths
can recur across contrasts. Bank SHA256:
`b6db8efdf34769e6b97aebc68486fdebe62c94b70da36b4be30335e10d10ba6b`.

At each embedding/block depth and token position, compute delta = h_on - h_off
for each pair. The LAT axis is the leading eigenvector of

    C_k = mean_pairs(delta delta^T).

This is PCA on the **symmetrized bag {+delta, -delta}**. Explicit symmetrization
keeps the consistent contrast signal: subtracting the mean from a bag of
uniformly oriented positive-minus-negative differences would instead remove
that signal. This convention is a specified implementation choice for this
experiment, not a claim that all LAT implementations make the same choice.
See the original [representation engineering paper](https://arxiv.org/abs/2310.01405)
and [PCA reader implementation](https://github.com/andyzoujm/representation-engineering/blob/main/repe/rep_readers.py)
for the broader method and its sign/centering conventions.

There is one axis per class, depth, and token. Multiple-token interventions
combine those per-token axes, as the earlier pipeline combined per-token mean
vectors. They are not one PCA over concatenated multi-token states.

## Turning axes into interventions

PCA fixes neither sign nor intervention magnitude. With unit LAT axis u_k and
the equal-weight mean of the 100 saved full-support class means, define

    q_k = u_k [u_k^T (mu_k - global_mean)].

Thus the direction comes from PCA, and its natural scale comes from projecting
the class displacement onto that axis. The expression is invariant to the
arbitrary eigenvector sign. To replace source class 24 by target k on the
existing held-out pairs, apply

    alpha (q_k - q_24).

Class 24 remains the source of the **intervention task**, not the negative class
used to estimate every LAT axis. This distinction is essential when comparing
inter-class geometry.

Location and strength are chosen on the original 256 training calibration
pairs per target, independently at each checkpoint. The scan covers embedding
and blocks 1–6, every individual token, tokens 2–5, and all six tokens; strengths
are 0.5, 1, and 2. Held-out examples never select axes, locations, or strength.
The training paths used for axis estimation can overlap calibration paths, as
the previous full-support means did; all held-out paths remain excluded.

Both Brier gain and actual edited endpoint accuracy are saved. Exact activation
patches, norm-matched random edits, full-support DoM at LAT-selected locations,
and means of the same contrast bag at those locations are retained as controls.
The original DoM baseline used strength 1; a separately selected strength-1 LAT
replay is included to make that comparison clearer. Unit-strength locations
are selected independently from the saved calibration grid restricted to alpha
1, then evaluated on the same fixed held-out pairs.

## Geometry metric

At the third token, separately for each fixed transformer block, compare all
100 unit LAT axes using mean absolute cosine over the 4,950 distinct unordered
class pairs. This is **mean absolute cosine, not RMS cosine**. No class-reference
subtraction, time smoothing, or model EMA is applied to this geometry.

Controls are (1) the mean of the same contrast bag and (2) the full-support class
means after subtracting the equal-weight global class mean, equivalent in
direction to class-versus-balanced-rest DoM. All three methods use the same
100 classes, depth, token, and cosine statistic.

## Results

At step 8000, mean LAT steering gain is 0.75337 and actual steering accuracy is
85.02%. At step 20000, mean gain is **0.99836**, minimum target gain is 0.98881,
and actual steering accuracy is **99.8918%** (24/32 targets have perfect measured
accuracy; worst target 98.7013%). Original DoM final gain is 0.99624. The strength
search differs: LAT selected alpha 1 for 18 targets and alpha 2 for 14 targets.
Do not attribute that small gain difference solely to PCA. LAT with strength
fixed at 1 reaches final mean gain **0.97945** and accuracy **98.8095%**.

LAT learns to steer later than the full-support DoM baseline over much of the
transition: at 6000 their respective mean gains are 0.30889 and 0.44986; at 8000,
0.75337 and 0.79641. The calibrated LAT and DoM means are both about 0.997 by
16000. Checkpoint dips remain; no scores are smoothed or made monotone.

Final mean absolute pairwise cosine:

| Block | LAT axes | Same-bag mean | Full-support DoM |
|---|---:|---:|---:|
| 1 | 0.14649 | 0.13752 | 0.13773 |
| 2 | 0.12418 | 0.11443 | 0.11469 |
| 3 | 0.11948 | 0.11136 | 0.11158 |
| 4 | 0.11715 | 0.10982 | 0.11001 |
| 5 | 0.11709 | 0.11042 | 0.11067 |
| 6 | 0.11958 | 0.11176 | 0.11201 |

LAT preserves strong steering but **does not reduce inter-class angular
overlap**. The difference is also present against the mean of the identical
contrast bag, so it is not explained simply by comparing 792 pairs with the
larger full-support mean estimate. PCA maximizes
contrast energy, which does not mathematically guarantee removal of surviving
parent information or orthogonality.

Early in training, the LAT axes are almost parallel across classes (mean
absolute cosine near 1 at 1000–2000). This is consistent with the leading
contrast component capturing variation common to many classes; these plots
alone do not identify which information causes it.

## Validation and execution

Three local tests passed: comparison with exact eigenvectors, constant/zero
contrasts, and sign-invariance of intervention scale. All 147 checkpoint/depth
fits passed eigen-residual checks and independent exact top-eigenvalue spot
checks. Maximum eigen residual is 6.49e-6, maximum eigenvalue spot-check error
8.35e-7. Maximum Brier generalization difference versus the previous bfloat16
evaluation is 0.002378; this replay uses FP32 forwards throughout. The precision
difference is another reason not to interpret tiny differences from the old
baseline as a clean estimator improvement.

Pilot jobs: **17567588.0** (8000, g197), **17567588.1** (20000, g191), both NVIDIA
H100, both successful, about 20 seconds per checkpoint. Full jobs:
**17567599.0–18**, exact step mapping in `submit_lat_full.sub`. Compute hosts were
g188, g191, g197, and g205, all H100s. Total measured checkpoint runtime was
419.74 GPU-job seconds; the independent audit added 16.1 seconds.

`audit_lat_replay.py`, Condor **17569242.0** on g187's H100, independently
recomputed all selected held-out edits using intervention hooks. Its gains and
accuracies match the cached-activation replay exactly at all 21 checkpoints.
It also evaluated independently selected strength-1 locations.

At the final checkpoint, the bank was split into two disjoint sets of contrast
pairs, four against each negative class (396 contrasts per half). Individual
paths can occur in both halves; this is a pairing-sensitivity check, not an
independent-dataset confidence interval. At the third token:

| Block | Median half-to-half absolute cosine | Minimum | Median half-to-full |
|---|---:|---:|---:|
| 1 | 0.99271 | 0.97212 | 0.99813 |
| 2 | 0.99824 | 0.98817 | 0.99958 |

## Reproduction and artifacts

The fixed contrast bank is made by `prepare_lat_bank.py`. `run_lat.sh` executes
`scan_lat.py` on allocated GPUs; `submit_lat_pilot.sub` and `submit_lat_full.sub`
specify all 21 original model checkpoints. Use fresh output directories for a
new experiment, rather than overwriting completed results.

To regenerate the downloaded plots locally:

       python3 plot_lat.py runs/emergence_lat_s46 \
         --means-run runs/emergence_large_locations_s46

Plotting requires 21 completed checkpoints by default and incorporates the
completed `audit.json`. The local `summary.json` describes the full trajectory.

Per-checkpoint directories contain `history.json` (calibration grid, selected
locations, test gains, true steering accuracies, controls, eigen diagnostics)
and `directions.pt` (axes, scaled class codes, means of contrast pairs, explained
energy, selected interventions). `contrast_bank.pt` contains the fixed path IDs.

`figures/steering_curves.csv` and `figures/geometry.csv` are directly readable
plotting tables. `figures/geometry.npz` contains all 100-by-100 cosine matrices
and explained-energy arrays, with axes checkpoint/block/class/class, blocks
1–6, and class IDs ordered as in the checkpoint histories. The accuracy and
control PNG/PDF files accompany the two headline plots. The explicitly named
`lat_pilot_final_geometry` figure is retained as a final-checkpoint bar chart.
