# Last-token mean steering: depth scan

Follow-up: an [independent intervention audit](MEAN_INTERVENTION_RESULTS.md)
reproduces successful analytic mean steering using target-minus-reference means
at tokens 2–5. It also reproduces the negative last-token result in FP32. The
depth scan below applies specifically to last-token, target-minus-rest edits.

Changing intervention depth does not rescue the current difference-of-means
edit. On the final checkpoints of seeds 46–48, all 144 latent/seed/depth
steering scores are negative (largest: -0.63021). There is **no supported depth
for this edit**. The exact last-token patch fails through block 5 and succeeds
at block 6, where the classifier directly reads the patched token.

Open the [depth comparison](runs/emergence_mean_depths_summary/depth_scan.png)
or [numeric table](runs/emergence_mean_depths_summary/depth_scan.csv).
These diagnostics supplement the earlier
[block-1 checkpoint replay](MEAN_STEERING_RESULTS.md); they do not replace its
held-out curves.

## Measurements

Mean raw target Brier skill across eight latents and three seeds, step 10,000:

| Post-block depth | Mean steering | Exact patch | Clean target |
|---|---:|---:|---:|
| 1 | -0.89877 | -0.89883 | 0.98312 |
| 2 | -0.89876 | -0.89876 | 0.98312 |
| 3 | -0.89878 | -0.89879 | 0.98312 |
| 4 | -0.89876 | -0.89873 | 0.98312 |
| 5 | -0.89879 | -0.88964 | 0.98312 |
| 6 | -0.89872 | 0.98312 | 0.98312 |

The random-direction control is also approximately -0.8987 at every depth.
Plots show raw scores without flooring or smoothing, with individual seed means
as faint lines. The block-6 patch matches each clean-target score to within
1.2e-16. Its success is an architectural check: after the final block, all
remaining computation reads only the last position. It does not demonstrate
a context-independent representation of the intermediate graph latent.

The scan changes only transformer depth. Graph layer remains 3, position remains
5 (the last token), and each 128-dimensional target-versus-rest vector is added
at strength 1. Means use all 800,000 distinct training paths, with equal weight
per alternative latent, FP32 forwards, and FP64 accumulation, exactly as in the
previous replay. No optimizer or strength sweep is used.

Evaluation uses the final 256 of the original 1,024 training calibration pairs
per latent. These paths also contribute to the means and were used to train the
model, so these are **calibration scores, not held-out generalization scores**.
No test scores were read to compare or choose depths. This explains why the clean
target and block-1 scores differ from the earlier held-out replay. Test path IDs
are checked only for split isolation.

## Implication

Do not select block 6 as a successful mean-steering depth merely because its
patch control works. The scan rules out a successful final-checkpoint edit at
any of the six depths under the current definition. It does not test every
depth throughout training, alternative reference subtraction, or other token
positions. Earlier-token information being reused downstream is consistent with
the patch results, but this scan does not isolate that mechanism.

The depth question is resolved for the current final-checkpoint protocol:
there is no viable choice. Further work would need to change or diagnose the
intervention definition, rather than assume that another depth fixes it. The
existing defaults and all previous experiment artifacts are preserved.

## Reproduction and verification

Code commit `7708e13`, synchronized to the clean cluster clone via a verified
Git bundle and fast-forward pull. GitHub push failed public-key authentication.

```bash
condor_submit_bid 2000 submit_emergence_depths.sub
python3 plot_emergence_depths.py runs/emergence_mean_depths_s*/depth_scan.json \
  --out-dir runs/emergence_mean_depths_summary
python3 -m unittest discover -s tests -v
```

Fresh output directories are required for repeat scans. Jobs **17553011.0–2**
completed with exit code 0 on **g197, g187, and g190**, respectively, all NVIDIA
H100 GPUs. Condor reports 18, 18, and 12 seconds of runtime. The wrapper launches
`scan_emergence_depths.py /fast/fdraye/toy_model_linear/emergence_wide_s46
--out-dir /fast/fdraye/toy_model_linear/emergence_mean_depths_s46`, likewise
s47/s48. Outputs were downloaded to the matching local `runs/` directories.

All 12 unit tests pass, including calibration split isolation and final-block
patch equivalence. All 18 saved direction tensors were verified exactly against
their saved FP64 class means, with shape `[8, 1, 128]` and finite values.
