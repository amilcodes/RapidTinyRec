# Profiling physical-prefix decoding

This experiment measures what remains after shared-prefix caching. The model,
FP32 math, native GQA, beam search and all five scored probabilities are unchanged.
There are no new performance results yet.

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
