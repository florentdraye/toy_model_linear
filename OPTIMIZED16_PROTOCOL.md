# Optimized steering replication: 16 lower-frequency latents

This experiment builds directly from the successful setup behind
`runs/emergence_optimized_summary/raw_fidelity.png`.

The architecture and training protocol are unchanged: width 128, six causal
transformer blocks, four heads, ReLU² MLPs, batch 2,048, AdamW learning rate
.001, weight decay .1, 100-step warmup, 10,000 steps, and measurements every
100 steps. The graph seed is 0, data seed 314, graph layer 3, frequency ratio
10, and reference latent 24. Seeds 46–55 vary initialization and training draws.

The intervention is after block 1 at token positions 2–5. For every latent and
checkpoint, the matched-pair mean displacement initializes 400 projected Adam
Brier-loss steps. A second 400-step log-loss optimization starts from that saved
vector. The iterate with the best Brier loss on the fixed validation quarter is
used. The vector norm cannot exceed the RMS natural matched activation change.
The 462 fixed held-out pairs are excluded from fitting and selection.

The original eight ranks were 1, 15, 29, 43, 57, 71, 85, 99. Ranks 1 and 15
are removed. Sixteen ranks evenly cover the retained range:

`29, 34, 38, 43, 48, 52, 57, 62, 66, 71, 76, 80, 85, 90, 94, 99`

They map to latents
`92, 41, 78, 79, 97, 28, 7, 54, 5, 69, 38, 90, 48, 26, 23, 44`, with training
probabilities from 1.298% down to 0.255%. Selection is fixed before new results.

`submit_optimized16.sub` launches ten H100 jobs. Each job completes source
training and log-loss refinement sequentially, avoiding dependency races.
Outputs are under `/fast/fdraye/toy_model_linear/emergence_optimized16_{source,refined}_sSEED`.
Run `python3 collect_optimized16.py` to fetch refined histories and reproduce
the archived figures locally under `runs/emergence_optimized16_summary/`.

The main estimator is checkpoint-wise, latent-wise averaging across model seeds.
The archived plot style applies a documented centered three-checkpoint display
smooth to the bounded headline and retains raw checkpoint dots. `raw_fidelity.png`
is the unclipped, unsmoothed seed average matching the user's reference image.
