# Profiling physical-prefix decoding

This experiment measures what remains after shared-prefix caching. The model,
FP32 math, native GQA, beam search and all five scored probabilities are unchanged.
The completed profile confirms matrix multiplication dominates. Decode attention
is the next substantial target; removing the remaining KV materialization alone
has a much smaller ceiling.

## Measured breakdown

These are summed GPU activity durations from actual graph-replay traces, in ms.
They include profiling effects and are not a partition of the earlier 126.697 ms
500-user median. One trace was taken per length.

| Work | 76 tokens | 101 tokens | 121 tokens |
|---|---:|---:|---:|
| Prefill, all GPU activities | 21.664 | 25.934 | 27.075 |
| Four decode forwards, all GPU activities | 84.133 | 86.081 | 87.546 |
| Decode matrix multiplications | 63.131 | 62.809 | 63.373 |
| Decode native attention | 16.867 | 18.670 | 19.171 |
| Decode fused cat/layout/RoPE kernels | 2.472 | 2.931 | 3.304 |
| Both vocabulary normalizations, five steps | 1.811 | 1.810 | 1.811 |
| Suffix reordering | 0.880 | 0.871 | 0.882 |
| Final unused cache reorder | 0.301 | 0.299 | 0.304 |
| Copy one physical prefix | 0.128 | 0.133 | 0.137 |

The rows for decode operations are components of the four-forward row, not
additional costs. Prefill plus decode matrix multiplications consume about
83.5–88.6 ms. Diagnostic traces attribute about 42–43 ms to decode MLP projections
and 11.1–11.3 ms to the full-vocabulary projection. The vocabulary projection is
retained because full-vocabulary normalization remains part of every score.

Generated code confirms some kernels with long fused names only load, select and
store prefix/suffix values. Other similarly named kernels also perform RoPE.
The 2.472–3.304 ms category therefore bounds more than pure dense KV copies;
removing all those copies cannot remove all the category's work. This makes a
copy-only change a small target compared with the 16.867–19.171 ms decode attention.

Actual replay kernel sums are 112.862/118.980/121.629 ms and activity unions are
113.971/120.146/122.798 ms. Instrumented request wall times are
158.256/140.883/143.275 ms. The larger first-trace overhead is retained, with no
claim about its cause. Exclusive host scope timings include waits and overlap GPU
execution; they cannot be added to these GPU durations or treated as removable
latency. In particular, the roughly 11–12 ms prefill host call does not imply an
11–12 ms saving from capturing it.

All 60 uninstrumented, 30 light-profile, six traced, ten diagnostic warmup and
30 normal warmup outputs match historical compiled GQA bit for bit: 6,800 rank
positions and sequence values, 34,000 transition values, maximum deltas zero.
Forty graph shapes stayed frozen with no measured compilation or capture.
Setup/warmup took 347.904 s, including nested backend compilation of 268.764 s,
graph preparation of 208.684 s, and diagnostic warmup of 1.416 s. These intervals
overlap. Uninstrumented per-length medians span 124.872–133.213 ms across six trials;
the ten uniformly sampled lengths are not the distribution of the frozen500.

Audit, trace-derived tables, per-request timing/hash rows and provenance are in
[benchmarks/profiles/shared-prefix](../benchmarks/profiles/shared-prefix).
The source is `278999c`, job 7879030 completed successfully in 6m51, and all 30
CPU/evidence checks passed before the run.

## Next experiment

The native attention trace executes 112 kernels, each handling a single decode
query position. Qwen2 has 12 query heads sharing two KV heads. The next pilot will
pack each six-head group into six query rows of one KV head before calling native
attention, then restore the original head layout. Each row still attends to the
same keys and values; no probability or token is dropped. Prefill remains unchanged.

This is a proposed layout optimization, not a verified speedup or exactness claim.
It applies only to unmasked, noncausal, single-token decode after RoPE, with zero
dropout. The mathematical mapping follows PyTorch's
[GQA head repetition semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).
Changing the native kernel's shapes may still alter floating-point rounding, so
the pilot must match every ranking and score bit before a 500-user comparison.

## Method

Warm all 40 decode graph shapes used by the frozen 500-user cohort. Select the
first user at each of its ten history lengths and run six uninstrumented sweeps,
reversing length order on alternate sweeps. These 60 samples are a representative
profile baseline, not another 500-user latency estimate. Compilation, graph
preparation and warmup are recorded separately and have overlapping intervals.

After timing, record light scopes for three sweeps and CPU/CUDA traces at lengths
76, 101 and 121. Actual graph replay is the primary trace. A separate diagnostic
launches the same compiled prefix forward using the same stable input buffers
without CUDA replay, exposing operation and shape correlations. It is warmed
before freezing compilation and is used only for attribution.

The independent audit checks every saved ranking, sequence score and transition
score against the historical compiled-GQA outputs, including warmup and traced
requests. It checks source hashes, graph counts, input ownership accounting and
zero measured compilation/capture. Generated Inductor sources and compressed
Chrome traces are retained. Fused concatenation/layout/RoPE kernel time must not
be labeled pure KV-copy time. Host durations overlap GPU execution; GPU activity
sums and unions are reported separately.

Run on one allocated A6000 with FP32 and TF32 disabled:

```bash
python -m rapidtinyrec.benchmark_shared_prefix \
  --release-root RELEASE --reference-results GQA_RESULTS --output-root OUTPUT \
  --full-cohort --phase 1 --profile-prefix
python -m rapidtinyrec.audit_prefix_profile OUTPUT/phase_1 \
  --historical GQA_RESULTS --snapshot SNAPSHOT/rapidtinyrec \
  --run-record RUN_RECORD --output AUDIT.json
```

Use fresh `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` directories. No next
optimization is selected until the measured breakdown is available.
