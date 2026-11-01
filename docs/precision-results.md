# Reduced-precision measurements

The default stack uses FP32 with TF32 disabled. The new folded entry point
accepts `--precision bf16-experimental` as an explicit quality tradeoff. TF32
remains research-only. The historical research runners and failed experiments
are not merged into the runtime path.

## Measured quality and latency

Both experiments use the frozen 500-user Office cohort, RTX A6000 and beam 50.
They retain the full vocabulary, both normalizations and the five scored
tokens. Latencies exclude model loading, compilation, capture and warmup.

| Experiment | Median, ms | p95, ms | HitRate@10 | NDCG@10 |
| :--- | ---: | ---: | ---: | ---: |
| FP32 control in TF32 study | 114.431 | 119.520 | 0.114 | 0.079595 |
| TF32 candidate | 82.298 | 83.645 | 0.114 | 0.079595 |
| TF32 control in BF16 study | 82.210 | 83.595 | 0.114 | 0.079595 |
| BF16 candidate | 56.532 | 57.645 | 0.114 | 0.079855 |

Each study has its own paired control. Do not compound gains or treat these
as a direct paired BF16-versus-FP32 measurement.

TF32 changes 142 of 500 full ranking lists, despite identical observed
HitRate@10 and NDCG@10 for every user. BF16 changes all 500 full ranking lists;
14,816 of 25,000 rank positions match its TF32 control. That position agreement
is not a recommendation-quality metric.

For BF16 versus TF32, HitRate@10 is 57/500 in both arms: one user gains a hit,
one loses a hit, and 498 are unchanged. The paired user-bootstrap 95% interval
for the difference is [−0.006, +0.006], or ±0.6 percentage points. NDCG@10
improves for four users, regresses for four, and is unchanged for 492. Its mean
change is +0.000259526, with interval [−0.001515240, +0.002011809].

The aggregate hides descriptive subgroup regressions. At prompt length 91
(85 users), HitRate@10 falls by 1/85 (1.176 percentage points) and NDCG@10 by
0.003401. At length 101 (42 users), NDCG@10 falls by 0.000970. The evidence does
not establish population equivalence or a zero quality cost; these intervals
describe uncertainty on this cohort, not a guaranteed loss bound for deployment.

## Experimental BF16 opt-in

The measured recipe loads model parameters in BF16 and retains BF16 model
logits and KV tensors. Hugging Face normalization, processor scores and beam
sequence scores remain FP32. The policy enables TF32 for remaining FP32
matmuls, disables cuDNN TF32 and BF16 reduced-precision reductions, preserves
the reviewed split-K setting, and disables reduced-precision math-SDPA
reductions. It is not simply an autocast switch, and uses no INT8/INT4 weights.

Use `python -m rapidtinyrec.run folded --precision bf16-experimental` with
`--release-root`, `--reference-results` and a new `--output-root`. Omit the
precision flag for FP32. Selection happens before loading, compilation and
capture; every launcher call creates a fresh process and compiler caches.
Runtime checks enforce model/logit/KV dtypes and FP32 owned scores. The process
restores its numerical policy on normal exit and exceptions, and records the
selected precision and restoration in output metadata. Global attention and
model hooks retain their context-managed restoration.

The default cohort is four users at the shortest/longest validated lengths,
with two passes in forward/reverse order. `--full-cohort` runs all 500 users
once; `--repeats` controls pilot repetitions only. `--save-full-scores` writes
lossless dense FP32 scores for acceptance auditing outside the latency timer,
and retains score tensors to verify their lifetime. It increases memory/I/O
cost and is not required for ordinary benchmark use.

The release acceptance uses the new default FP32 command and BF16 opt-in in
separate processes, with same-policy historical token/sequence/transition
references, BF16 ancestry references, all five full scores, reverse-order
repeats, immutable parameter/input/table digests, and compiler/capture freeze
checks. Its small cohort is an integration check, not a new quality study or
replacement for the full500 chart.

An experimental opt-in can expose the measured tradeoff with these limits.
Changing the default or promising a quality-loss budget would need an explicit
acceptance threshold and representative held-out evaluation. The existing
[single-worker and shape restrictions](release-scope.md#supported-use) still
apply; this does not create a concurrent serving API.

## Saved evidence

- [TF32 allocated recovery audit](https://github.com/amilcodes/RapidTinyRec/blob/065d4dbb2346403820adba751a8efb5067dd3d5a/research/tf32-full500-recovery/evidence/recovery-audit.json): original generation completed; the initial audit failed on a catalog path. A separate allocated saved-output audit passed without regenerating results. The initial failure remains preserved.
- [BF16 corrected full500 independent review](https://github.com/amilcodes/RapidTinyRec/blob/7de96b5caa7741a99d13bc9cb0ef859bdd992936/research/bf16-full500-results/evidence/bf16-full500-corrected-independent-review.json): execution, source/protocol, dispatch, stability and allocated numerical audits passed; no automatic promotion or quality-equivalence claim.
- [BF16 policy](https://github.com/amilcodes/RapidTinyRec/blob/7de96b5caa7741a99d13bc9cb0ef859bdd992936/rapidtinyrec/bf16_policy.py) and [full500 benchmark](https://github.com/amilcodes/RapidTinyRec/blob/7de96b5caa7741a99d13bc9cb0ef859bdd992936/rapidtinyrec/benchmark_bf16_full500.py) define the measured precision and output contract.

The README chart numbers are rounded from these saved full500 reports. The
opt-in acceptance is separate; no historical job is reopened or re-audited.
