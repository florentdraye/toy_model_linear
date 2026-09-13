# Reproducing successful analytic mean steering

The earlier successful mean control was real. A fresh independent implementation
reproduces it using full-support target-minus-reference means at block 1, token
positions 2–5, with strength 1 and no optimization. The FP32 held-out target
Brier skill is **0.95873**, with **97.36% target accuracy**, averaged over eight
latents and three seeds. All 24 latent/seed skills exceed 0.80.

This is explicitly a four-token intervention. The requested last-token-only
variant remains unsuccessful in these models. The earlier simplification changed
two ingredients at once: token coverage and the negative mean population.

![Token location and mean definition](runs/emergence_intervention_audit_summary/intervention_diagnosis.png)

## Controlled comparisons

Final checkpoints, graph layer 3, transformer block 1, strength 1. Each mean uses
the complete distinct training-path support for its class, with FP32 forward
passes and FP64 accumulation. Held-out scores use the original 462 matched pairs
per latent, with reference latent 24 and identical continuation edge choices.

| Edited positions | Mean subtraction | Held-out Brier skill | Target accuracy |
|---|---|---:|---:|
| Last token (5) | Target minus rest | -0.87071 | 7.27% |
| Last token (5) | Target minus reference | -0.87069 | 7.27% |
| Tokens 2–5 | Target minus rest | -0.51439 | 22.37% |
| Tokens 2–5 | Target minus reference | **0.95873** | **97.36%** |

Changing either ingredient alone does not recover the earlier performance at
unit strength. Replacing the earlier optimized directions was not, by itself,
the source of failure: the archived unoptimized matched-mean control already
scored 0.95786. A fresh matched-pair mean using only the first 768 training pairs
scores 0.95777. The full-support class-mean translation scores 0.95873 without
matching calibration contexts, endpoint supervision, or fitting an optimizer.

Target-minus-rest inserts a contrast against an average of 99 alternative
classes. The evaluation input always belongs to the specific reference class.
Target-minus-reference instead implements the corresponding class-mean
translation. These are distinct interventions in a categorical latent space.

## Independent implementation and location diagnosis

`audit_emergence_interventions.py` runs in FP32, explicitly captures the full
post-block state, edits it, and continues through the remaining blocks. It
does not rely on the measurement hook for its scores. At all six depths in all
three seeds, both clean continuation and edited continuation agree exactly with
the original forward/hook implementation on the checked examples. A unit test
also compares both implementations' steering and patch metrics. The negative
last-token result is reproduced without BF16.

Exact-patch heatmaps show that patchable information generally reaches token 4
before it reaches the last token. At blocks 3–5, patching token 4 is effective;
patching token 5 is ineffective until block 6. Token 4 patches contain
context-dependent information, so their success alone does not establish a
constant graph-layer-3 steering direction there. The location used by the model
also differs across seeds at earlier blocks, explaining why a fixed single
early token is less reliable than the four-position slice.

The diagnostic sweep covers all six post-block depths, every single position,
the suffix and all-token layouts, three analytic mean definitions, and strengths
1, 2, 4, 8, and 16. Calibration uses the last 256 original training pairs per
latent; those paths also enter the full-support class means. No held-out scores
select the sweep winner. The block-1 held-out comparisons above were fixed in
the audit script before execution. No last-token configuration in this sweep
has a positive seed-averaged calibration skill, even at the larger strengths.

[Last-token strength curves](runs/emergence_intervention_audit_summary/last_token_strength.png)
retain these raw negative scores. The audit does not prove impossibility for
other models, intermediate sub-block sites, or other intervention definitions.

## Checkpoint replay

`remeasure_emergence_means.py --site suffix --contrast reference` replays the
full-support target-minus-reference edit across the 101 saved checkpoints per
seed. The edit is computed anew at each checkpoint. The original last-token,
target-minus-rest defaults remain available and unchanged; the successful
comparison must be requested explicitly. The old artifacts are preserved.

![Full-support mean steering and generalization](runs/emergence_reference_means_summary/steering_gain.png)

Replay directions have shape `[8, 4, 128]`; each position has its own class-mean
translation. All model parameters are frozen. The 800,000-path training support
excludes both sides of every held-out evaluation pair. Original generalization
curves are copied exactly and independently checked by a new forward pass.
The figure uses the existing three-checkpoint moving average, raw dots, seed
bands, and explicit flooring of negative scores. Unclipped results are available
in [raw_fidelity.png](runs/emergence_reference_means_summary/raw_fidelity.png).

All **303 replay snapshots** were verified exactly against their saved FP64
class means. Every saved generalization score is exactly unchanged; independent
forward-pass discrepancies are also zero. The replay uses the original BF16
evaluation convention and ends at mean steering skill **0.95834**, compared with
**0.98525** for generalization. The smallest individual final steering skill is
0.79599. These differ slightly from the independent FP32 audit above. The mean
steering curves are noisier than the optimized curves and do not establish exact
simultaneous emergence with generalization.

## Reproduction

```bash
condor_submit_bid 2000 submit_emergence_interventions.sub
condor_submit_bid 2000 submit_emergence_reference_means.sub
python3 plot_emergence_interventions.py runs/emergence_intervention_audit_s*/audit.json \
  --out-dir runs/emergence_intervention_audit_summary
python3 plot_emergence.py runs/emergence_reference_means_s*/history.json \
  --out-dir runs/emergence_reference_means_summary --smooth-window 3
python3 -m unittest discover -s tests -v
```

Fresh run directories are required. Audit code commit: `e1ff837`; replay code:
`bbc18da`. Both were synchronized to a clean cluster clone with verified Git
bundles and fast-forward pulls. GitHub remains unsynchronized after the earlier
public-key authentication failure.

Audit jobs **17553073.0–1** completed on **i104/i208**. Job 17553073.2 failed in
the GPU inventory command on i207 before the Python audit started; its retry
**17553082.0** completed on **g201**. All used NVIDIA H100 GPUs. Audit outputs:
`/fast/fdraye/toy_model_linear/emergence_intervention_audit_s46`, likewise s47/s48.

Replay jobs **17553101.0–2** completed successfully on H100 nodes **g192, g205,
g201** in 51, 57, and 48 seconds. Command: `run_emergence_means.sh` with the
source wide run, refined baseline history, `--site suffix --contrast reference`.
Outputs: `/fast/fdraye/toy_model_linear/emergence_reference_means_s46`, likewise
s47/s48; local copies use the corresponding `runs/` paths. Thirteen unit tests pass.
