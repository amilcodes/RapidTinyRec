# Native FP32 backend diagnostic: rejected

The corrected diagnostic completed all 12 fixed-input cases on one RTX A6000. Explicit cuBLAS matches default exactly. cuBLASLt is repeatable and its API dispatch is verified, but **all 12 down-projection cases change FP32 bits and are slower in both median and p95**. Gate/up outputs match exactly; their small timing changes do not provide consistent p95 benefit. No generation pilot, full500, workspace/algorithm sweep or baseline promotion follows this result.

Measured source `aa9ba129ed0dc4020bbd8184a9918dd1c9dd01c8`, job `7956300`, run `fp32-backend-diagnosis-retry1-20261026T043139Z-8a353d2d043c`. The job completed in 45 seconds (diagnostic wall time 32.649 s). The exact saved auditor exited 1 because candidate equality failed, while integrity, explicit-cuBLAS control, within-backend stability and dispatch checks passed. No audit assertion was weakened. All 47 submitted files, 246 raw files and 144 content-addressed tensor blobs were checked; 1,728 comparison rows, 108 timing rows and 36 dispatch rows are retained.

## Exactness and performance

Original-input observations across four users × three layers are shown below. Changed-row-order and restored-input probes have the same aggregate equality outcome. They are repeated observations, not additional users.

| Projection | Original cuBLASLt bit matches / values | Max absolute delta | Median of 12 case medians, default → Lt (µs) | Median of 12 case p95 values, default → Lt (µs) | Lt faster median / p95 cases |
|---|---:|---:|---:|---:|---:|
| gate | 5,376,000 / 5,376,000 | 0 | 124.660 → 124.391 | 125.761 → 125.615 | 12/12 / 6/12 |
| up | 5,376,000 / 5,376,000 | 0 | 124.706 → 124.321 | 125.632 → 125.465 | 12/12 / 7/12 |
| down | 63,210 / 921,600 | 3.814697266e-05 | 138.206 → 155.199 | 139.514 → 156.279 | 0/12 / 0/12 |

For down, 858,390 of 921,600 original-input values differ; including changed/restored states, 2,575,170 of 2,764,800 differ. Maximum absolute delta is 0.00003814697265625. Gate/up each match all 16,128,000 observed values across the three states. Explicit cuBLAS matches all 35,020,800 versus-default observations. All 490,291,200 within-backend repeat, graph, graph-repeat, retained, restored and after-timing observations match, with zero deltas and no nonfinite outputs. These are projection values, not generated sequence/transition scores.

Each timing sample averages 32 frozen graph replays; there are eight alternating-order rounds per role/backend. These column summaries are medians across case statistics, **not a pooled request-latency p95**. cuBLASLt is faster in median for 24/36 cases and p95 for 13/36; all down cases regress. Explicit cuBLAS control case medians are 124.350/124.355/138.054 µs for gate/up/down, close to default. The full timing CSV retains dispersion and every round. Staging, backend switching, capture/warmup, profiling/API logging, extraction and file writes are outside these microtimings. No compilation or capture occurs during timing. No inference speedup, rankings, SID hit/NDCG/MRR or population quality equivalence was measured.

## Dispatch and arithmetic

All 24 default/explicit-cuBLAS probes log three `cublasSgemm_v2` calls; all 12 cuBLASLt probes log three public `cublasLtMatmul` API calls, with no cuBLAS SGEMM fallback. Both paths use FP32 `COMPUTE_32F`; TF32 and tunable-operation search remain disabled. The installed wheel matches the corrected pin `08187d9e0fba026dc8217405802ab5381dc88d90`, PyTorch 2.14.0+cu126 and Transformers 4.55.4.

The cuBLAS SGEMM path itself logs internal Lt machinery. Therefore “cuBLAS” versus “Lt” identifies the requested/public dispatch path, not exclusive internal libraries. Gate/up use `ampere_sgemm_128x64_tn`, grid `[70,1,3]`, 128×64 tiles and three split-K partitions on both paths. Down changes from `ampere_sgemm_128x64_tn`, grid `[12,1,16]`, plus a separate reduction kernel, to `ampere_sgemm_64x32_sliced1x4_tn`, grid `[24,2,12]`. API logs identify 16 partitions with `REDUCTION_SCHEME_COMPUTE_TYPE` versus 12 with `REDUCTION_SCHEME_INPLACE`. This directly supports a changed reduction organization; proprietary arithmetic internals are not exposed.

The untouched native workspace preferences differ: cuBLAS 8,519,680 bytes versus Lt 1,048,576 bytes. The SGEMM down implementation logs 4,915,200 workspace bytes; this is an observed default-path difference, not a workspace-tuning experiment. No follow-up workspace search is authorized by this result.

## Setup, memory and boundaries

Model parameters occupy 6,176,878,592 logical and unique bytes. Each case owns three independent input buffers and nine independent output buffers: 14,080,000 logical/unique bytes, excluding model, original activations/weights and private graph pools. Input shapes are `[50,1536]` for gate/up and `[50,8960]` for down; independent original down inputs prevent upstream rounding from confounding the comparison. The source saves original, reversed-row, restored and retained outputs and validates ownership on one sequential CUDA stream.

Timing-phase allocated/reserved bytes are 6,245,720,064 / 6,356,467,712 in every case, with equal timing-phase peaks. After activation collection they were 6,222,505,984 / 6,949,961,728; the allocated increase while a case is live is 23,214,080 bytes. Earlier allocated peak was 6,770,070,528 and maximum observed reserved peak 6,973,030,400. Model load is 1.502 s; activation collection 1.759 s; case setup ranges 1.563–1.744 s, including nested capture/warmup. These intervals must not be summed into a purported independent setup total. The review retains every case's counters and memory peaks.

Slurm accounting confirms four allocated/requested CPUs, 32 GiB and one GPU. The raw site prolog prints “CPUs: 1”; `resource-accounting.txt` preserves the authoritative allocation counters separately without editing that log.

## Preserved failure and reproduction

The first job 7955758 stopped before measurements because the build guard incorrectly pinned the release tag rather than the wheel's parent commit. Its original files and failed audit remain in the separate startup-failure archive. The official comparison changes only one test import; all five reviewed numerical source files are identical. The measured repair changed the build pin and incomplete-audit diagnostic only, leaving GPU operations and exactness checks unchanged.

All complete archive members are preserved in four SHA-256-verified shards below 48 MiB, including raw outputs, logs, profiler traces, frozen source/config, run record, exact saved audit argv/cwd, independent review/review script, submission guard, CI log, exporter/tests and prior startup-failure evidence. Extract **all** shards into one directory, then run from `source-snapshot`:

```bash
python3 -m rapidtinyrec.audit_fp32_backend ../raw/outputs/diagnosis \
  --snapshot . --run-record ../run-record.json --output ../reproduced-audit.json
```

Expected: integrity/control/stability/dispatch true; candidate exact false; exit 1. A failed equality gate is the result, not an export error.

## Research decision

Keep the accepted folded baseline unchanged. The bounded backend candidate is exhausted, with no generation follow-up. Combined with the inexact wider/batched MLP, exact but unhelpful constraint replay, and beam replay's small aggregate win with per-length regressions, the tested approaches have not produced a new improvement meeting every current gate.

Exact score fidelity and recommendation quality remain distinct. Allowing different FP32 accumulation could expand candidates but would require a new, explicit quality/ranking tolerance; this slower down path would still be unattractive. Native FlashRec's Qwen3 BF16/FP8 and per-codebook scoring require a separate quality/latency comparison with agreed held-out SID metrics and matched device/users/catalog/timing/concurrency. It cannot establish a speedup for our Qwen2 FP32 full-vocabulary contract. No precision, scoring or acceptance criterion changes have been made. Reassess the next research objective with the user before another GPU tuning experiment.
