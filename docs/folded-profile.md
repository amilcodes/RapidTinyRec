# Profile the folded-query baseline

The last audited 500-user comparison measured 114.841 ms median and 119.898 ms
p95 on an RTX A6000, with exact FP32 rankings and scores. This experiment profiles
that implementation to choose the next matrix-multiplication optimization.
This profile changes no inference code and makes no new speedup claim.

The model, cache, attention wrapper and scoring sources are unchanged from the
passing folded-query experiment. A dedicated entry point retains all 40 graph
shapes and uses ten representative users, one per history length. Six timed
sweeps precede profiler initialization. Three light sweeps and six CPU/CUDA
traces cover actual replay and an uncaptured compiled-forward diagnostic at
76, 101 and 121 tokens. The diagnostic uses the same stable input layouts and
folded attention; its timings are for attribution only.

The audit groups recognized linear operations by their matrix dimensions:

| Input width → output width | Attribution |
|---|---|
| 1536 → 8960 | MLP gate or up projection |
| 8960 → 1536 | MLP down projection |
| 1536 → 1536 | Query or attention output projection |
| 1536 → 256 | Key or value projection |
| 1536 → 152265 | Full-vocabulary projection |

Identical shapes remain grouped; shape information cannot distinguish gate from
up, query from output, or key from value. Actual replay provides the serving
kernel profile; diagnostic traces expose operation/shape correlations hidden
inside CUDA graph launches. Correlated GPU activity counts are not asserted to
equal operation calls. Inspect generated Inductor code before attributing fused
work. Host time, activity sums and activity unions overlap and must not be added.

The independent audit checks every saved rank, sequence and transition FP32 bit
against historical compiled GQA, the pinned release/cohort and submitted source
hashes. It checks 40 frozen graphs, ownership/storage accounting and zero measured
compilation/capture. Full-vocabulary normalization and all five probabilities
remain intact. Setup, compiler work, graph preparation and warmup are recorded
separately from inference; their intervals are nested. The ten representatives
are not a new 500-user latency distribution.

```bash
python -m rapidtinyrec.benchmark_folded_profile \
  --release-root RELEASE --reference-results GQA_RESULTS --output-root OUTPUT \
  --phase 1 --full-cohort
python -m rapidtinyrec.audit_folded_profile OUTPUT/phase_1 \
  --historical GQA_RESULTS --snapshot SNAPSHOT/rapidtinyrec \
  --run-record RUN_RECORD --output AUDIT.json
```

Run in a fresh A6000 process with empty variant-specific Inductor/Triton disk
caches, FP32 and TF32 disabled. A candidate selected from the profile must pass
a bounded paired pilot before the frozen 500-user comparison. Preserve failed
exactness checks, setup costs and latency regressions. No tolerance substitution.

## Audited result

Job 7889882 completed in 6m59s on one A6000. Source `dba5327` and all 52
submitted files were frozen. All 136 saved outputs matched historical compiled
GQA: 6,800 rank positions, 6,800 sequence FP32 values (217,600 bits) and 34,000
transition values (1,088,000 bits), with zero numerical deltas. All 40 graphs
remained frozen; no measured compilation or capture occurred.

The 60 uninstrumented representative requests measured **117.420 ms median /
120.382 ms p95**. This is not a replacement for the frozen 500-user result.

| Actual replay GPU activity (ms) | 76 tokens | 101 tokens | 121 tokens |
|---|---:|---:|---:|
| Decode matrix multiplication | 62.997 | 62.750 | 62.610 |
| Prefill matrix multiplication | 20.324 | 24.236 | 25.295 |
| Decode attention | 4.551 | 5.106 | 5.279 |
| Decode fused cat/layout/RoPE | 2.659 | 3.123 | 3.489 |

Diagnostic shape attribution puts decode MLP gate/up at **27.788–27.933 ms**,
MLP down at **14.554–14.850 ms**, full vocabulary projection at **11.013–11.216
ms**, Q/output projections at **6.678–6.993 ms**, and K/V projections at
**2.480–2.621 ms**. These diagnostic measurements are not additional time to add
to the replay table. Prefill gate/up costs another 11.276–12.436 ms in replay.

Generated Inductor code calls two separate `extern_kernels.mm` operations on
the same MLP input, followed by fused SiLU/multiply and the down projection.
The trace includes SGEMM and split-K reduction kernels; multiple GPU activities
per linear operation explain why activity counts exceed operation counts.
Fused cat labels also contain layout and rotary operations, so their entire time
cannot be attributed to KV copying.

The next candidate is packing gate/up output columns into one decode GEMM.
It targets the largest attributed group, but the changed GEMM shape may select
a different accumulation path. A paired pilot must prove bit equality and an
end-to-end benefit before the 500-user run. Weight storage, setup peaks, output
slices and any copies must be included in its accounting.

Setup was **363.876 s**, with nested backend compilation **273.237 s**, graph
preparation **222.430 s**, and diagnostic warmup **1.295 s**. These intervals
overlap and must not be summed. Inference allocated **7.485 GiB**, reserved
**9.791 GiB**, and peaked at **7.779 GiB allocated**.

Evidence: [summary](../benchmarks/profiles/folded-query/summary.json),
[audit](../benchmarks/profiles/folded-query/audit.json),
[source and raw hashes](../benchmarks/profiles/folded-query/source.json),
[request timings](../benchmarks/profiles/folded-query/requests.csv).
