# Six-hour FP32 A6000 research window — October 24, 2026

The research window ended at 15:40 UTC (11:40 Eastern); new submissions closed at 15:25 UTC. The folded-query profile is complete and published. The packed-MLP pilot never started and was cancelled at closeout with zero compute time. No additional latency improvement was validated in this window, and no new 500-user candidate comparison ran.

## Verified profile

The [folded-query profile](folded-profile.md), [PR #12](https://github.com/amilcodes/RapidTinyRec/pull/12), ran on one A6000 as job `7889882` for 6m59s. Measured source `dba532787923a3a3ea03ee355815605d17a14d9d`, evidence `4711e463321deadb835ac8cc25c1dfbf41d86108`; all 52 submitted files and raw output hashes were preserved.

All 136 saved outputs match historical compiled GQA exactly: 6,800 rank positions, 6,800 FP32 sequence values (217,600 bits) and 34,000 transitions (1,088,000 bits), with zero deltas. All 40 graphs were frozen; measured compilation and capture were zero. Sixty uninstrumented requests across ten representative users measured 117.420 ms median and 120.382 ms p95. These representatives do not replace the frozen 500-user distribution.

| GPU activity group | Measured range (ms) | Evidence scope |
|---|---:|---|
| Decode matrix multiplication | 62.610–62.997 | Actual replay |
| Prefill matrix multiplication | 20.324–25.295 | Actual replay |
| Decode attention | 4.551–5.279 | Actual replay |
| Decode fused cat/layout/RoPE | 2.659–3.489 | Actual replay, mixed work |
| Decode MLP gate/up | 27.788–27.933 | Diagnostic shape attribution |
| Decode MLP down | 14.554–14.850 | Diagnostic shape attribution |
| Decode vocabulary projection | 11.013–11.216 | Diagnostic shape attribution |
| Decode Q/output projections | 6.678–6.993 | Diagnostic shape attribution |
| Decode K/V projections | 2.480–2.621 | Diagnostic shape attribution |

Diagnostic categories subdivide decode work; do not add them to replay totals. Activity counts can include SGEMM and split-K reduction for one linear operation. Fused labels contain layout/rotary work as well as copies. Instrumented traces, GPU sums/unions, and host scopes overlap and must not be added to uninstrumented latency.

Profile setup was 363.876 s, containing nested backend compilation 273.237 s, graph preparation 222.430 s, and diagnostic warmup 1.295 s. These overlap. Inference allocated 7.485 GiB, reserved 9.791 GiB, and peaked at 7.779 GiB allocated. See the [profile evidence](../benchmarks/profiles/folded-query/summary.json) and [independent audit](../benchmarks/profiles/folded-query/audit.json).

## Selected candidate and measurement limit

Generated Inductor code confirms two gate/up GEMMs on the same input. The [packed-MLP candidate](packed-mlp.md), [draft PR #13](https://github.com/amilcodes/RapidTinyRec/pull/13), combines these into one wider decode GEMM, followed by gate/up views. Prefill retains separate projections. Both variants use the validated folded-query attention and physical-prefix cache.

The implementation packs weights once after loading and makes original gate/up parameters disjoint contiguous views into that allocation. For the pinned model, its ownership invariant is 3,082,813,440 unique gate/up bytes and zero persistent duplicate bytes. This is a source/CPU-checked invariant; measured GPU memory and setup costs remain pending. The wider GEMM may change rounding, so mathematical equivalence and CPU tolerance do not establish score-bit equality.

All 46 CPU/evidence tests and CLI checks passed [CI 40100234692](https://github.com/amilcodes/RapidTinyRec/actions/runs/40100234692) at source `9b94664b1800260674d63a26cdc33eb5a4cb5370`. The single 30-minute A6000 pilot was submitted at 13:57 UTC: job `7893602`, run `packed-mlp-pilot-20261024T135738Z-d85533e2a0f0`. All 56 submitted files are immutable and hash-verified. It remained PENDING (Resources) at the deadline and was cancelled through a controller-side PENDING-state filter. Slurm accounting confirms CANCELLED with 00:00:00 elapsed compute time. The fetched outputs and logs directories are empty; submission files and all source hashes are preserved. There are no candidate GPU rankings, score comparisons, median/p95, setup or memory measurements to report. This is a queue limitation, not a numerical rejection. No full-cohort job was submitted because the exactness and latency gate was never reached.

The exact saved audit command exited nonzero because phase_0/manifest.json does not exist. Source integrity was independently verified, but output exactness was never evaluated. The [unrun summary](../benchmarks/unrun/packed-mlp/summary.json) and [source/submission hashes](../benchmarks/unrun/packed-mlp/source.json) preserve that distinction. Optimization and submissions are closed; the research automation is being removed after publication. No running allocation was interrupted.

## Baseline and remaining work

The previously validated [folded-query 500-user result](folded-query-heads.md) remains the performance baseline: native-prefix median/p95 127.198/133.628 ms versus folded-query 114.841/119.898 ms in that same allocation. All 25,000 rank positions, 25,000 sequence values and 125,000 transitions match exactly against paired controls and historical compiled GQA. Allocated memory is 7.485 GiB for both. Cold setup increased 346.870→369.764 s; artifact-reuse setup increased 148.152→157.549 s. This result predates the current research window and is not compounded with the representative profile timings.

The largest measured remaining target is the decode MLP gate/up group, followed by MLP down and full-vocabulary projection. Packing is implemented but has no measured GPU outcome yet. Native FP32 GEMM library/layout choices and compiler tuning remain untested hypotheses. Any further work requires a paired exact pilot before a same-source 500-user comparison. The prior combined-suffix experiment changed score bits and remains ineligible.

All runs retain FP32, TF32 disabled, the pinned Office checkpoint, frozen cohort, beam50, all five probabilities, and both full-vocabulary normalizations. No merge, retraining, paid infrastructure, subagent work, or alternate-GPU substitution occurred.
