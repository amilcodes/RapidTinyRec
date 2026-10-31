# Validated FP32 stack

PRs #7–11 promote decode graph replay/direct KV reorder, shared graph input
storage, physical-prefix KV storage, the supporting profile, and folded-query
attention. The measured inference implementation is unchanged by promotion.
The existing folded benchmark is the entry point; this repository does not yet
expose a general concurrent serving API.

## Result and numerical baseline

On 500 frozen Office users, the final paired comparison measured:

| Metric | Physical prefix + native GQA | Folded-query attention |
| :--- | ---: | ---: |
| Median generation latency | 127.198 ms | 114.841 ms |
| p95 generation latency | 133.628 ms | 119.898 ms |
| Allocated GPU memory | 7.485 GiB | 7.485 GiB |

Every one of 25,000 rank positions, 25,000 sequence scores and 125,000 transition
scores matches both the paired control and historical compiled GQA bit for bit.
That equality is measured on this cohort and pinned runtime. Earlier shared
prefill and compilation changed some score bits relative to the original
released beam loop; promotion does not redefine that original comparison.
Different paired experiments have different absolute baselines; do not add
percentage gains or subtract unrelated runs to estimate a combined speedup.

The input-arena dependency is a memory win with a small latency regression:
29.16 → 18.78 GiB allocated, with median/p95 +0.188/+0.349 ms in its own run.
Physical-prefix storage subsequently reduced allocated memory to 7.485 GiB in
its paired comparison. All regressions and historical incomplete runs remain
in the evidence archive.

## Supported use

- Pinned Office Qwen2 checkpoint; PyTorch 2.14.0+cu126, Transformers 4.55.4,
  tokenizers 0.21.4; Linux, RTX A6000, four CPU threads.
- FP32 weights and arithmetic, TF32 disabled, one unpadded request, beam 50,
  all five scored tokens, and both full-vocabulary normalizations retained.
- Empty-cache prefill followed by one-token cached decoding with full attention.
  The measured history lengths are the ten cohort lengths between 76 and 121.
- One model and sequential CUDA stream per isolated worker. Attention dispatch
  is patched process-wide while its context is active. Do not run other model
  requests concurrently in that process or reuse borrowed graph outputs after
  replaying their shape. The benchmark consumes outputs within that lifetime.
- Warm and capture all required shapes, then freeze the compiler and graph pool.
  Uncaptured shapes after freeze raise an error; there is no general fallback
  for padded batches, sliding attention, or arbitrary multi-token decoding.

Cold setup/warmup was 369.764 s for the folded path; with disk compiler artifacts
reused it was 157.549 s. Processes, GPU allocations and graph captures were fresh
in both cases. These setup costs are excluded from generation latency and are
not a cold-request latency claim. Allocated memory was 7.485 GiB; reserved memory
was 9.313–9.779 GiB and peak allocated memory during inference 7.779–7.784 GiB.

## Run and evidence

Use the existing [folded-query phase commands](folded-query-heads.md) with the
pinned release and historical GQA reference outputs. The command is
`python -m rapidtinyrec.benchmark_folded_prefix`; the four ABBA phases compare
the control and folded path in fresh processes. It is a reproducible benchmark,
not an online request-serving command. Promotion does not require rerunning old
GPU experiments; the published source hashes and completed audits remain the
measurement evidence.

[All measurements](benchmarks.md) retain per-length/half results, setup and
memory limits. [The full folded report](folded-query-heads.md) links the original
pilot/full500 provenance and audit. The input-arena, prefix and replay ablations
remain available for reproduction.

## Excluded from this promotion

TF32 and BF16 precision branches, combined suffix scoring, packed MLP/QKV,
constraint/beam-helper replay, concurrent serving, prompt caching, beam waves,
and later buffer-reuse or score-capture experiments are not merged here.
The approximately 82 ms TF32 and 57 ms BF16 medians reported in later research
use different numerical policies; they are not results for this FP32/TF32-off
release. Precision/quality tradeoffs require a separate decision.
