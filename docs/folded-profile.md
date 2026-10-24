# Profile the folded-query baseline

The last audited 500-user comparison measured 114.841 ms median and 119.898 ms
p95 on an RTX A6000, with exact FP32 rankings and scores. This experiment profiles
that implementation to choose the next matrix-multiplication optimization.
No new speedup is claimed before the run and audit complete.

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
