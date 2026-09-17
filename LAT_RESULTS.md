# LAT steering and inter-latent geometry

## Status

Implemented and submitted a 21-checkpoint replay of the **original**, unaveraged
width-512 seed-46 model, at steps 0, 1000, ..., 20000. All 100 graph-layer-3
latent classes get LAT axes; the existing 32 targets retain their fixed 462
held-out intervention pairs. No training or model execution runs on the Mac.

Two pilot checkpoints, 8000 and 20000, are complete and downloaded. The remaining
19 jobs were submitted as Condor cluster **17567599**; five were still pending
or running at the last successful queue check. The Mac then lost DNS resolution
for `login.cluster.is.localnet`. Both configured DNS servers return NXDOMAIN.
The full trajectory is **not yet verified or collected**. An additional replay
audit and unit-strength evaluation are implemented but not yet submitted.

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
The original DoM baseline used strength 1; therefore a separately selected
strength-1 LAT replay is implemented to make that comparison clearer.

## Geometry metric

At the third token, separately for each fixed transformer block, compare all
100 unit LAT axes using mean absolute cosine over the 4,950 distinct unordered
class pairs. This is **mean absolute cosine, not RMS cosine**. No class-reference
subtraction, time smoothing, or model EMA is applied to this geometry.

Controls are (1) the mean of the same contrast bag and (2) the full-support class
means after subtracting the equal-weight global class mean, equivalent in
direction to class-versus-balanced-rest DoM. All three methods use the same
100 classes, depth, token, and cosine statistic.

## Pilot observations (not the full trajectory)

At step 8000, mean LAT steering gain is 0.75337 and actual steering accuracy is
85.02%. At step 20000, mean gain is **0.99836**, minimum target gain is 0.98881,
and actual steering accuracy is **99.8918%** (24/32 targets have perfect measured
accuracy; worst target 98.7013%). Original DoM final gain is 0.99624. The strength
search differs: LAT selected alpha 1 for 18 targets and alpha 2 for 14 targets.
Do not attribute that small gain difference solely to PCA.

Final mean absolute pairwise cosine:

| Block | LAT axes | Same-bag mean | Full-support DoM |
|---|---:|---:|---:|
| 1 | 0.14649 | 0.13752 | 0.13773 |
| 2 | 0.12418 | 0.11443 | 0.11469 |
| 3 | 0.11948 | 0.11136 | 0.11158 |
| 4 | 0.11715 | 0.10982 | 0.11001 |
| 5 | 0.11709 | 0.11042 | 0.11067 |
| 6 | 0.11958 | 0.11176 | 0.11201 |

In this pilot, LAT preserves strong steering but **does not reduce inter-class
angular overlap**. Finite-pair estimator stability will be checked with two
balanced halves of the contrast bank in the pending audit. PCA maximizes
contrast energy, which does not mathematically guarantee removal of surviving
parent information or orthogonality.

## Validation and execution

Three local tests passed: comparison with exact eigenvectors, constant/zero
contrasts, and sign-invariance of intervention scale. The two pilot jobs passed
all per-depth eigen-residual checks and independent exact top-eigenvalue spot
checks. Maximum eigen residual is 8.97e-7, maximum eigenvalue spot-check error
7.15e-7. Maximum Brier generalization difference versus the previous bfloat16
evaluation is 0.000833; this replay uses FP32 forwards throughout.

Pilot jobs: **17567588.0** (8000, g197), **17567588.1** (20000, g191), both NVIDIA
H100, both successful, about 20 seconds per checkpoint. Full jobs:
**17567599.0–18**, exact step mapping in `submit_lat_full.sub`.

The additional `audit_lat_replay.py` independently recomputes held-out edits
using intervention hooks, selects strength-1 locations from saved calibration
scores, and checks final axes on balanced split-half contrast banks. It requires
all 21 completed results before submission with `submit_lat_audit.sub`.

## Reproduction and resumption

1. Restore normal access to the cluster public hostname.
2. Check the cluster clone is clean. Sync local commit `6768cf6` or its descendant
   before submitting the audit; the cluster was last at `749ffe8`.
3. Verify all 21 `step*/history.json` files are complete and inspect failed-job
   logs if any. Do not overwrite existing completed output directories.
4. Submit `condor_submit_bid 2000 submit_lat_audit.sub` after syncing the code.
5. Download outputs to `runs/emergence_lat_s46/` and run:

       python3 plot_lat.py runs/emergence_lat_s46 \
         --means-run runs/emergence_large_locations_s46

Plotting requires 21 completed checkpoints by default. `--expected-checkpoints 2`
was used only to inspect the pilot, not to represent a completed trajectory.
The local summary currently describes those two pilot checkpoints.
