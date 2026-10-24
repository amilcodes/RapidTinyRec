# Fold query heads into attention rows

The four-user A6000 pilot cuts median latency by 10.05% and p95 by 11.16%,
with identical FP32 rankings and scores. Allocated memory is unchanged.
The 500-user comparison is submitted; these are pilot results only.

The shared-prefix profile measures native decode attention at 16.9–19.2 ms per
request, compared with 2.5–3.3 ms for the entire fused cat/layout/RoPE category.
This experiment targets attention.

Qwen2 has 12 query heads and two KV heads. For a single decode token, query
heads 0–5 attend to KV head 0 and heads 6–11 to KV head 1. After RoPE, reshape
the contiguous query from `[50, 12, 1, 128]` to `[50, 2, 6, 128]`, call native
SDPA with matching query/KV head counts, then reshape the output back.

Each query row has exactly the same key/value rows as before. Rows are independent
because this specialization has no mask, no causality and no dropout. It changes
the launch geometry, not the mathematical probabilities. It does **not** prove
bit equality: kernel shape changes can alter FP32 rounding. The A6000 pilot
checks that separately. The head mapping follows
[PyTorch's GQA semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

The wrapper leaves prefill, masked attention, causal attention, dropout and
non-GQA calls unchanged. Both variants retain the physical shared-prefix cache,
temporary dense attention K/V, native FP32 attention, beam50, all five token
probabilities and both full-vocabulary normalizations. This is not a custom
shared-prefix attention kernel. The context patches SDPA process-wide and belongs
in an isolated inference worker.

## Validation

Pilot job `7880867` completed in 18m18s on one RTX A6000. Measured source:
`af2c05dae10a6b107616da964eb2a1360eb5cde1`. Independent integrity and exactness
audits pass. The 24 paired requests cover four users and six trials each.

| Pilot measurement | Native GQA + physical prefix | Folded query heads |
|---|---:|---:|
| Median latency | 128.487 ms | 115.579 ms |
| p95 latency | 133.926 ms | 118.976 ms |
| Allocated GPU memory | 6.142 GiB | 6.142 GiB |
| Reserved GPU memory | 7.418–7.660 GiB | 7.418–7.660 GiB |
| Peak allocated during inference | 6.436–6.442 GiB | 6.436–6.442 GiB |
| Logical graph-input bytes | 79,599,936 | 79,599,936 |
| Unique graph-input bytes | 15,546,688 | 15,546,688 |

All **1,200/1,200 rank positions**, **1,200/1,200 sequence FP32 values**
(38,400 bits) and **6,000/6,000 transition FP32 values** (192,000 bits) match.
Maximum sequence and transition differences are zero. Both variants also match
the historical compiled-GQA outputs, including all 48 warmup outputs; repeat
stability and input/output ownership checks pass. All eight graphs per process
were frozen before measurement, with zero measured compilation or capture.

All 24 paired requests were faster by 11.223–16.003 ms; the median paired
difference is −13.193 ms. At 76 tokens, median/p95 changes from
124.970/125.491 to 112.403/113.430 ms; at 121 tokens, from
133.060/134.173 to 118.465/119.110 ms. Both ABBA halves improve.
This small, repeated extreme-length sample does not establish the 500-user
distribution or variance across independent allocations.

| Phase | Compiler artifacts | Setup | Nested backend compilation | Nested graph preparation/warmup |
|---|---|---:|---:|---:|
| 0: native | Cold | 344.398 s | 267.553 s | 208.947 s |
| 1: folded | Cold | 360.289 s | 273.872 s | 223.048 s |
| 2: folded | Reused from phase 1 | 151.523 s | 69.834 s | 86.144 s |
| 3: native | Reused from phase 0 | 142.692 s | 66.443 s | 79.699 s |

These intervals overlap; do not add them. All remain outside inference. Folding
increased setup time in this pilot. The per-phase audit separately retains setup
memory peaks. Temporary dense K/V buffers still exist: there is no memory saving
or measured claim here that attention itself became a particular factor faster.

Evidence: [summary](../benchmarks/pilots/folded-query-heads/summary.json),
[independent audit](../benchmarks/pilots/folded-query-heads/audit.json),
[paired timings and output hashes](../benchmarks/pilots/folded-query-heads/requests.csv),
[source and artifact hashes](../benchmarks/pilots/folded-query-heads/source.json).
The identical-source full run is job `7882772`,
`folded-full-20261024T040658Z-73d312250079`. Its results are pending.

## Protocol

CPU tests cover head mapping, inverse layout, zero-copy contiguous query reshape,
fallback dispatch, context restoration and full-graph Dynamo tracing. CPU
mathematical comparisons use a tolerance and are not an FP32 GPU equality claim.

The A6000 pilot uses four users at the shortest and longest histories, six paired
trials, eight warmed graphs, and four isolated processes in native/folded/folded/
native order. Each process has fresh model/GPU allocations. Phases0/1 start with
cold compiler artifacts; phases2/3 reuse same-variant disk artifacts but capture
fresh graphs. The planned full comparison uses the frozen500 users split by
even/odd cohort index and warms all40 shapes per phase. Both jobs are bounded at
30 minutes, with the full run gated on a passing pilot and useful measured gain.

Dense-layout warmups use each phase's attention implementation. They are labeled
`dense_forward`, not historical references. The independent audit compares every
timed and warmup output with the historical compiled-GQA results, and pairs the
two variants by user/repeat. Setup, compilation, graph preparation and warmup
stay outside inference and have overlapping intervals. ABBA phase pairing does
not measure adjacent request alternation or run-to-run variance.

```bash
python -m rapidtinyrec.benchmark_folded_prefix \
  --release-root RELEASE --reference-results GQA_RESULTS --output-root OUTPUT \
  --phase PHASE
python -m rapidtinyrec.audit_folded_prefix OUTPUT \
  --historical GQA_RESULTS --snapshot SNAPSHOT/rapidtinyrec \
  --run-record RUN_RECORD --output AUDIT.json
```

Run each phase in a fresh process. Use fresh variant-specific Inductor/Triton
directories for phases0/1 and `--reuse-compiler-artifacts` in phases2/3. Add
`--full-cohort` only after the complete pilot audit passes. Exactness failures and
regressions remain evidence; the auditor saves them and returns a failure status.
