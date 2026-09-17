# Width-512 DoM replication with dense measurements

Protocol fixed before the first new runs: model seeds **46–55**, including all three
previous seeds. After the provisional ten-seed latent curves remained visibly
noisy, the replication was extended prospectively to seeds **56–95**, for 50
model seeds overall. Every run uses 20,000 training steps and measurements every **100** steps (201
checkpoints). No seed selection, weight EMA, curve smoothing, clipping,
time alignment, or normalization by each curve's final value.

Preserves the original seed-46 model in `emergence_lat_s46/figures/lat_steering.png`:
width 512, six blocks, four heads, ReLU², batch 2048, AdamW learning rate .001,
100-step warmup, weight decay .1. Graph seed 0, data seed 314, graph layer 3,
same 32 frequency-ranked targets, frequency ratio 10, reference class 24.
Seeds vary model initialization and training draws; the graph, split, frequency
assignment, mean support, calibration pairs and held-out pairs remain fixed.

At each checkpoint, `mu_target - mu_reference` uses every distinct training
path in each relevant class, FP32 forwards and FP64 sums. Unused classes are
skipped because they do not enter either mean. Strength is exactly 1.
Scan the same 56 intervention locations: embedding and six blocks, each with
six single tokens, the latent suffix, or all tokens. Choose separately for each
latent/checkpoint using the last 256 of the fixed 1,024 training pairs. Ties go
to the first cell in depth/token order. Calibration examples overlap mean
support, but neither includes held-out paths.

Both paths in each of the **462 fixed held-out pairs per target** are outside
training support. Generalization scores the real target input; steering scores
the reference input edited toward the target, against the target endpoint.
Both use `1 - mean(||p - one_hot(y)||²) / .99`. Negative values are retained.
Actual target/steering accuracy and exact activation-patch scores at the DoM
selected location are separate fields. All new evaluations use FP32, whereas
the archived DoM curves used BF16 test forwards. Training remains BF16.

Run through `condor_submit_bid 2000 submit_dom_multiseed.sub`; the executable is
`bash run_dom_multiseed.sh SEED`. Outputs:
`/fast/fdraye/toy_model_linear/dom_multiseed_sSEED/`. Each contains history,
calibration scores, selected locations, mean-support counts/hashes, graph,
pair IDs, selected vectors at every checkpoint, model snapshots every 1,000
steps, and a resumable optimizer/sampler checkpoint. Re-run with `--resume`
to recover an interrupted seed. Local raw histories and plots are collected
under `runs/dom_multiseed_summary/`.

Submitted as **17571550.0–9**, followed by the noise-reduction extension
**seeds 56–95**. The first runs started on g198/g205 with NVIDIA
H100 GPUs. Fetch results and generate plots with `python3 collect_dom_multiseed.py`.
The collector waits for measurements from all ten seeds, then plots only their
common checkpoints. Incomplete trajectories are prominently provisional.

`plot_dom_multiseed.py` writes the 32 mean steering/generalization curves,
all individual seeds, a 32-latent PDF atlas, and raw `curves.csv` / `curves.npz`.
The overview uses pointwise 95% percentile intervals from 4,000 bootstrap
resamples of whole model seeds (fixed bootstrap seed 914). The paired gap uses
the same seed resamples. Atlas bands instead show ±1 across-seed standard
deviation. Neither interval includes graph or evaluation-bank uncertainty.
NPZ arrays are `[seed, checkpoint, latent]`; the JSON summary names axes and
records bank hashes and per-seed final metrics. CSV has one row per
seed/checkpoint/latent, including selected depth/token positions.

The primary aggregation gives every seed equal weight. Uncertainty describes
model-seed variation with the graph and evaluation bank held fixed. Individual
seed curves must accompany means; weak runs are retained. This protocol tests
whether averaging yields stable emergence curves; it does not assume steering
must match generalization.

Validation: tiny CPU models check full-support means against the established
estimator, omission of unused classes, independent hook-based steering replay,
test-independent location selection, and unchanged RNG/model state.
