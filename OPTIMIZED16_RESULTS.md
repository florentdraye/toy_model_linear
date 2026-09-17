# Optimized steering with 16 lower-frequency latents

The requested replication is complete. It uses the exact optimized-direction
pipeline behind `runs/emergence_optimized_summary/raw_fidelity.png`, removes the
two highest-frequency ranks, fills the retained frequency range with 16 latents,
and averages checkpoint-wise across ten independently trained model seeds.

The primary result is
[`raw_fidelity.png`](runs/emergence_optimized16_summary/figures/raw_fidelity.png):
unclipped Brier skill, no temporal smoothing, one curve per latent, averaged over
seeds 46–55. The matching presentation figure is
[`steering_gain.png`](runs/emergence_optimized16_summary/figures/steering_gain.png):
raw dots, a centered three-checkpoint display smooth, and descriptive seed
uncertainty. Both have 101 checkpoints through step 10,000.

Final scores across 160 seed × latent measurements:

| Metric | Overall mean | Range of 10-seed latent means | Individual minimum |
|---|---:|---:|---:|
| Optimized steering Brier skill | 0.998317 | 0.984553–0.999995 | 0.857696 |
| Generalization Brier skill | 0.978948 | 0.950983–0.996060 | 0.884365 |
| Exact patch at the same site | 0.968132 | 0.896044–0.995567 | 0.393548 |
| Unedited target accuracy | 0.987054 | 0.968615–0.997403 | 0.939394 |

Every seed-averaged latent curve crosses 0.5 for both steering and
generalization. Steering crosses between 8 and 1,079 steps earlier for all 16
curves in this run. This is a descriptive outcome of the supervised vector
optimization; no timing alignment or score normalization was imposed.

The raw data are in
[`curves.csv`](runs/emergence_optimized16_summary/figures/curves.csv) and
[`curves.npz`](runs/emergence_optimized16_summary/figures/curves.npz). NPZ score
arrays use `[seed, checkpoint, latent]`; coordinate arrays name all dimensions.
[`summary.json`](runs/emergence_optimized16_summary/figures/summary.json) records
completion, coordinates, the held-out-bank hash, and final summaries. Each raw
history is under `runs/emergence_optimized16_summary/seed_SEED/history.json`.

Condor cluster **17571882.0–9** ran the ten seeds on allocated H100 nodes. Every
job performed source training and log-loss refinement sequentially and exited
with code 0. Cluster outputs remain under
`/fast/fdraye/toy_model_linear/emergence_optimized16_{source,refined}_sSEED`.
The full predeclared protocol and latent ranks are in
[`OPTIMIZED16_PROTOCOL.md`](OPTIMIZED16_PROTOCOL.md).
