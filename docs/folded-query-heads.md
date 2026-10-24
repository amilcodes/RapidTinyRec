# Fold query heads into attention rows

Experimental; no A6000 result yet. The shared-prefix profile measures native
decode attention at 16.9–19.2 ms per request, compared with 2.5–3.3 ms for the
entire fused cat/layout/RoPE category. This experiment targets attention.

Qwen2 has 12 query heads and two KV heads. For a single decode token, query
heads 0–5 attend to KV head 0 and heads 6–11 to KV head 1. After RoPE, reshape
the contiguous query from `[50, 12, 1, 128]` to `[50, 2, 6, 128]`, call native
SDPA with matching query/KV head counts, then reshape the output back.

Each query row has exactly the same key/value rows as before. Rows are independent
because this specialization has no mask, no causality and no dropout. It changes
the launch geometry, not the mathematical probabilities. It does **not** prove
bit equality: kernel shape changes can alter FP32 rounding. The A6000 pilot must
check that separately. The head mapping follows
[PyTorch's GQA semantics](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

The wrapper leaves prefill, masked attention, causal attention, dropout and
non-GQA calls unchanged. Both variants retain the physical shared-prefix cache,
temporary dense attention K/V, native FP32 attention, beam50, all five token
probabilities and both full-vocabulary normalizations. This is not a custom
shared-prefix attention kernel. The context patches SDPA process-wide and belongs
in an isolated inference worker.

## Validation

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
