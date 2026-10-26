# Packed MLP score diagnosis

The completed pilot shows no observed ranking-quality decrease in four users, but fails exact FP32 scores. The completed fixed-input diagnostic isolates mismatches to the projections: two separate GEMMs over packed weight views remain exact, while a wide GEMM and the batched pair differ. The batched alternative is also slower. Neither qualifies for a new end-to-end pilot or full500.

## Current evidence

The original implementation changes two FP32 projections, each 1536→8960, into one 1536→17920 projection. The packed weights retain the original bits; the gate/up parameters are contiguous row views of the same allocation. The generated Inductor code preserves the SiLU/product expression and down projection. Its split operation changes addressing, not the values. Prefill remains unchanged, and the first-token transition scores match exactly; differences begin in decode.

The fixed-input experiment confirms that changing projection organization changes the output bits before SiLU. Each output still sums 1536 products, but changing the output width can change the GPU launch and accumulation order. FP32 addition is not associative. Different reduction organization is the supported explanation; proprietary kernel internals were not inspected. This is not evidence of reduced precision, changed weights, or dropped probabilities. PyTorch explicitly does not promise bitwise equality between mathematically equivalent batched/sliced matrix multiplications: [numerical accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

All six repetitions per user were stable within each variant. Consequently a deterministic-algorithm setting would not by itself fix this cross-implementation mismatch. NVIDIA's [cuBLAS reproducibility guarantee](https://docs.nvidia.com/cuda/archive/12.6.1/cublas/index.html#results-reproducibility) concerns repeat execution under specified conditions, not equality between different GEMM shapes.

## Ranking quality

The independent sample has four users. Six repeats yield 24 paired requests and 1,200 matching rank positions, but do not increase the number of independent quality observations. All top-50 token sequences and their order match. The held-out target is a semantic ID; these metrics do not disambiguate item collisions.

| Metric | Folded control | Packed | Change |
|---|---:|---:|---:|
| Hit@1 / @5 / @10 / @20 | 0.25 | 0.25 | 0 |
| Hit@50 | 0.50 | 0.50 | 0 |
| NDCG@50 | 0.2952606492 | 0.2952606492 | 0 |
| MRR@50 | 0.2555555556 | 0.2555555556 | 0 |

The largest sequence-log-probability delta is 0.000030517578125; the largest transition delta is 0.000019073486328125. The sequence delta bounds the corresponding multiplicative probability change by about 0.003052%. The smallest observed adjacent top-50 score gap is 0.000164031982421875. That margin is reassuring for these outputs only: the excluded rank-51 and intermediate beam-pruning margins were not recorded. Near-ties elsewhere could change ranking. Four users do not support a cohort-wide quality-equivalence conclusion.

The pilot's paired median/p95 generation-call latency is 115.6900/119.7311 ms for the folded control and 111.3289/115.2957 ms for packed: reductions of 3.77%/3.70%. These timings include work inside generation, including output splitting. Input preparation, output extraction and file writes are excluded. Setup/packing/compiler/capture are separate. Exact-score equality failed, so no full500 comparison was submitted.

## Bit-exactness investigation

Returning to the original two projections restores the validated implementation; it also gives up the demonstrated one-wide-GEMM optimization. The tested single batched call containing the two original-shaped projections does not preserve bits and takes about 66% longer in projection microtimings. A custom or explicitly selected GEMM implementation would need to reproduce the baseline reduction order, not merely compute a more accurate answer. Higher precision is not a way to reproduce the baseline's FP32 rounding. This experiment does not prove that every possible backend/layout implementation must fail.

Job 7949846, run `packed-mlp-diagnosis-20261026T002626Z-67a7da628839`, completed in 65 seconds on A6000 `gpu1503`, exit 0:0, on October 26 at 00:53:37 UTC. Accounting confirms four allocated CPUs and 32 GB; the job had a ten-minute limit. All 47 frozen submitted files match. The unchanged package comes from source `9b94664b1800260674d63a26cdc33eb5a4cb5370`. PyTorch 2.14.0+cu126, CUDA 12.6, FP32 and TF32 off.

All 12 fixed-input cases (four users × layers 0/13/27) completed. Each projects 50×1536 activations. Both separate-GEMM variants match all 5,376,000 gate values, 5,376,000 up values, 5,376,000 SiLU/product values and 921,600 down-output values. Graph outputs match direct outputs in all 43,008,000 checked projection values. Identical values in contiguous versus split layouts produce identical SiLU/products in all 5,376,000 values.

For both wide and batched variants, gate/up/product/down matching counts are 262,310 / 241,755 / 155,154 / 28,944, out of those respective totals. Maximum absolute differences are 2.765655517578125e-05 / 2.384185791015625e-05 / 7.62939453125e-05 / 9.1552734375e-05. Identical error summaries do not prove the two variants' tensors equal one another; that pairwise comparison was not recorded.

The first-case profiler gives a more precise explanation than kernel names alone. Both separate and wide calls use the name `ampere_sgemm_128x64_tn`, but separate projections each launch grid `[70,1,3]` with a preceding 280-byte memset. Wide launches `[140,1,1]` without that memset. This is consistent with a change in split-K/reduction organization; the exact internal reduction tree is not exposed by the trace. Batched uses `ampere_sgemm_128x128_tn`, grid `[70,1,2]`, and a 256-thread block instead of 128 threads.

| Projection variant | Median of 12 case medians, µs | Exact versus original |
|---|---:|---|
| Separate original GEMMs | 244.0895 | Yes |
| Separate GEMMs over packed views | 245.2177 | Yes |
| One wide GEMM | 219.5500 | No |
| Batched two GEMMs | 406.0050 | No |

Each case contains eight timing samples, each averaging 32 CUDA-graph replays, with reversed variant order on alternating rounds. Wide is faster and batched is slower in every case median. Separate packed views do not give a consistent useful improvement. Per-case medians/p95/ranges and all eight samples are preserved; the first round is noticeably faster in many cases and is not discarded. These are projection microtimings, not generation-call latency or independent user observations.

Diagnostic current allocated/reserved memory is 6,572,664,832 / 7,176,454,144 bytes; lifetime peaks are 6,744,970,240 / 7,308,574,720 bytes. These include the diagnostic model, saved activations and probe allocations, and are not per-variant inference memory. Activation collection took 5.8341 seconds; model loading, packing, compilation and graph capture were not separately timed in this diagnostic. Original pilot setup and logical/unique prefix/suffix capacities remain in the pilot report.

Activations come from eager folded-attention/shared-prefill generation, not saved intermediate tensors from the original compiled pilot. Raw activations and projection tensors were compared on the GPU but not persisted; the evidence preserves input hashes, exact counts/deltas, kernel traces and timings. The successful diagnostic isolates a reproducible numerical cause on representative inputs without claiming replay of the original compiled intermediate values.

The next candidate is graph replay of unchanged constraint operations, preserving both full-vocabulary normalizations, all five scored probabilities and the existing beam selector/tie rules. This changes launch overhead, with staging and buffer ownership included, rather than revisiting the rejected wide or batched projections.

## Reproduction and evidence

The [diagnostic summary](../benchmarks/diagnostics/packed-mlp/summary.json) contains every case's median/p95/range, exact counts and launch grids. The [audit](../benchmarks/diagnostics/packed-mlp/audit.json), [source hashes](../benchmarks/diagnostics/packed-mlp/source.json), [archive manifest](../benchmarks/diagnostics/packed-mlp/archive-manifest.json) and [raw archive](../benchmarks/diagnostics/packed-mlp/evidence.tar.gz) retain all raw outputs/logs/traces, 47 frozen source files plus submission config, job/resource/run records, original auditor and invocation, source-analysis scripts, and 120 generated compiler files from the original pilot. The archive has 188 individually hashed files. Existing negative pilot evidence remains unchanged.

CPU-only evidence verification recomputes aggregates and timing statistics from saved on-device comparison records and checks all hashes:

```bash
python -m unittest discover -s tests -p test_packed_mlp_diagnosis.py -v
```

To inspect or reproduce the diagnostic, extract the archive and use `snapshot/diagnose_packed_mlp.py` with its pinned package and `snapshot/jobs/diagnose-packed-mlp.sh`. Its arguments are `--release RELEASE --historical GQA_RESULTS --output NEW_OUTPUT`; only run on an authorized A6000 allocation. `snapshot/oscar.packed-mlp-diagnosis.json`, `raw/job.sh` and `provenance/resources.json` record the exact resources/environment. `provenance/audit-command.json` preserves the exact original audit invocation; `provenance/audit_packed_diagnosis.py` uses the original workspace layout. The CPU evidence test above verifies the portable archive without that layout or a model/GPU.

The earlier [pilot report](packed-mlp.md) retains generation-call latency by length/half, original numerical failures, setup timings and memory/storage ownership. The [accepted folded-query baseline](folded-query-heads.md) remains the performance reference. Diagnostic microtimings do not update it.
