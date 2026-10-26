> Historical pilot report, published at c6af461. The full500 submission mentioned below has since completed; see [full result](beam-replay.md).

# Beam-state helper replay: exact pilot, modest latency benefit

The four-user pilot preserved every rank and FP32 score and qualifies for one same-source 500-user validation. Generation-call median fell **116.235044 → 115.725942 ms (0.438%)** and p95 **121.173444 → 119.515060 ms (1.369%)** against its paired folded control. This is a small pilot result, not a new accepted performance baseline. Only 16 of 24 paired requests were faster; one length/half had a negative median paired-request saving. The 500-user comparison is already running as job **7954755**, run `beam-replay-full-20261026T030606Z-5721a23f41b7`.

Pilot job **7953928**, run `beam-replay-pilot-20261026T022316Z-54e1dcd97c50`, completed in 18:51 on gpu1503, one NVIDIA RTX A6000, four CPUs, 32 GB, regular gpu/norm-gpu, with a 30-minute limit. Measured source is [`9bd2a86b5b90594caea8a235b6c5ed56ada8a369`](https://github.com/amilcodes/RapidTinyRec/tree/9bd2a86b5b90594caea8a235b6c5ed56ada8a369), which passed 53 CPU/evidence tests and CLI checks in [CI40258692706](https://github.com/amilcodes/RapidTinyRec/actions/runs/40258692706). All 49 submitted files/config are hash-verified and frozen. The full500 submission is byte-identical, including the original auditor; postprocessing uses the separately frozen corrected auditor described below.

The isolated candidate invokes the original bound HF 4.55.4 `_get_running_beams_for_next_iteration` and `_update_finished_beams`, retaining their ATen arithmetic and 100/150-candidate top-k. It starts from accepted folded/profile PR12. Vocabulary-wide selection, both full-vocabulary normalizations, SID constraints, all five scored probabilities, original HF saved-score clones, stopping synchronization and model/cache behavior are unchanged. FP32 and TF32-off are unchanged. The rejected packed-MLP and constraint variants are not included.

Instance wrappers enforce an explicit generation lease on one sequential CUDA stream. Every dynamic tensor is staged before replay; helper outputs are independent by length, step and method. Sequence capacity remains prompt+8 for five generated tokens. Final outputs and transitions are consumed before lease close. Pilot phases contain eight model graphs and 20 helper graphs; full500 requires 40 and 100. All captures are frozen before timing.

## Paired generation latency

Fresh processes run folded/beam/beam/folded ABBA. Four frozen users at lengths 76/121 each have six trials: 24 paired observations, not 24 independent users. Phases 0/1 start with cold compiler artifacts; phases 2/3 reuse corresponding disk artifacts with fresh model/GPU allocations and capture. ABBA pairs phases, not adjacent requests or independent runs. This allocation does not measure run-to-run variance.

| Stratum | Pairs | Folded median / p95 (ms) | Beam median / p95 (ms) |
|---|---:|---:|---:|
| Overall | 24 | 116.235044 / 121.173444 | 115.725942 / 119.515060 |
| Length 76 | 12 | 112.942366 / 114.168636 | 112.215905 / 113.207584 |
| Length 121 | 12 | 119.362524 / 121.286117 | 118.882211 / 119.632832 |
| Phase half 0 | 12 | 115.968285 / 120.988092 | 115.577763 / 119.632832 |
| Phase half 1 | 12 | 116.436223 / 120.846322 | 115.725942 / 119.172205 |
| Half 0, length 76 | 6 | 113.035724 / 113.751211 | 111.949133 / 112.616647 |
| Half 0, length 121 | 6 | 119.362524 / 121.127106 | 119.218629 / 119.709135 |
| Half 1, length 76 | 6 | 112.831211 / 114.175295 | 112.705039 / 113.321075 |
| Half 1, length 121 | 6 | 119.671229 / 121.114987 | 118.616559 / 119.182952 |

Median and p95 are lower in each aggregate stratum. However, half 0/length 121 has only 2/6 faster individual pairs and median paired saving **−0.089228 ms**. The modest consistent aggregate reduction and 3.18 MB allocated-memory increase justify one broader validation; they do not support a statistical significance claim. The current paired control is primary. Do not compound this with previous speedups.

The timer covers the synchronized generation call, including all input staging, wrappers and original HF saved-score clones. It excludes input preparation, output/transition extraction, post-request diagnostic scans and file writes. It is not total application request latency.

## Exactness, lifetime and quality

All **1,200/1,200 rank positions, 1,200/1,200 sequence FP32 values and 6,000/6,000 transition values** match the paired control bit for bit, with maximum sequence and transition deltas **0**. Both variants separately match historical compiled GQA for all timed outputs. All 48 warmup outputs match historical GQA; 16 warm repeated comparisons and 20 timed repeated comparisons per variant are exact.

Seven GPU probe scenarios cover equal scores, ties crossing kth positions, signed zeros, near ties, infinities, finished flags and nonzero length penalty. Four mixed-length changed-input requests per scenario exercise five steps and both helpers. Across both candidate phases, **3,248,000/3,248,000 values match in each category**: graph/direct, original direct-repeat, and retained outputs after five steps. All finite deltas are zero; integer sequence/ancestry and boolean values are checked alongside float bits. Raw compressed direct, graph and repeated outputs are preserved. The 5,880 exported helper-output comparisons include all three categories, with no dropped failures.

Every candidate request retains 35 distinct helper output storages and five independent original HF saved-score clones. Final sequences, scores and ancestry match retained final helper outputs for all 20 warm/timed requests per candidate phase: 109,500/109,500 values per phase. Staging counters, pinned helper hashes, source-derived shapes/strides/scalars, mixed lengths, request leases and frozen graph counts pass. There is **zero measured compilation or capture**.

Four unique users have unchanged held-out SID metrics in both variants. Hit/NDCG/MRR at 1, 5, 10 and 20 are each 0.25. At 50, hit is 0.5, NDCG 0.2952606492 and MRR 0.2555555556. Every paired change is zero. These are SID metrics without item-collision disambiguation. Four users cannot establish population quality equivalence, and repetitions do not increase the independent-user count.

## Memory and setup

The following are allocator bytes, measured separately in fresh phase processes. Inference peaks reset after setup. Reserved memory includes allocator/private-pool capacity, so it is not interchangeable with logical tensor storage.

| Phase | Variant | Allocated | Reserved | Inference peak allocated | Inference peak reserved | Setup peak allocated | Setup peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Folded | 6,595,040,256 | 7,964,983,296 | 6,911,101,440 | 7,964,983,296 | 8,411,338,752 | 9,219,080,192 |
| 1 | Beam | 6,598,222,336 | 8,009,023,488 | 6,914,229,248 | 8,009,023,488 | 8,411,338,752 | 9,202,302,976 |
| 2 | Beam | 6,598,222,336 | 8,269,070,336 | 6,920,076,288 | 8,269,070,336 | 7,533,031,936 | 8,269,070,336 |
| 3 | Folded | 6,595,040,256 | 8,225,030,144 | 6,916,948,480 | 8,225,030,144 | 7,529,849,856 | 8,225,030,144 |

Beam adds **3,182,080 allocated bytes** and **44,040,192 reserved bytes** in both paired halves. Helper inputs total 2,223,510 logical bytes and outputs 888,500; unique input/output storage is **3,112,010 bytes**. That minimum excludes graph intermediates and private-pool slack. Staged bytes per request are 886,755 at length 76 and 1,336,755 at length 121. The original five HF score clones total 152,265,000 bytes in both variants.

Model logical and unique parameter storage remain 6,176,878,592 bytes, including 3,082,813,440 gate/up bytes. Model graph inputs total 79,599,936 logical bytes and 15,546,688 unique bytes: shared prefix capacity 6,938,624, suffix capacity 8,601,600 and metadata 6,464 bytes. Prefix capacity is batch 1/121 tokens; suffix capacity is batch 50/3 tokens. Dense exact-length attention K/V remains temporary per layer. Model replay output suffix storage is 57,344,000 bytes.

| Phase | Model load (s) | Combined setup (s) | Backend compilation (s) | Model capture/warmup (s) | Helper capture/warmup (s) |
|---|---:|---:|---:|---:|---:|
| 0 | 1.743241 | 351.509608 | 269.294336 | 213.834429 | 0.000000 |
| 1 | 1.737835 | 367.737418 | 276.144474 | 218.004324 | 0.119484 |
| 2 | 1.734577 | 157.783176 | 68.487824 | 85.287186 | 0.130075 |
| 3 | 1.732280 | 152.991543 | 70.883373 | 85.306398 | 0.000000 |

Combined setup includes probe execution/serialization, compilation, warmup and capture. These intervals nest/overlap and **must not be added**. Model loading is separate. Synthetic probes use separate graph pools; their captures are not the 20 generation helper graphs. No weight packing occurs in this candidate.

## Preserved audit correction and reproduction

The exact saved pilot audit was executed first and stopped on a structural stride assertion, before writing its result. The synthetic probe starts finished state with `sequences[:, :50].clone()` and the analogous ancestry slice of 100-candidate tensors. Clone preserves the singleton batch stride: `[8400,84,1]` for `[1,50,84]`, `[12900,129,1]` for `[1,50,129]`, and `[800,8,1]` for `[1,50,8]`. The original auditor incorrectly demanded canonical batch strides. Actual generation and later recurrent probe state use canonical strides.

Audit-only correction [`34b517a6a465f097f4d68e083eb1741cd9637009`](https://github.com/amilcodes/RapidTinyRec/commit/34b517a6a465f097f4d68e083eb1741cd9637009) requires exactly these source-derived layouts only for synthetic finished-helper step 0. It does not exempt arbitrary singleton strides or change score/index equality. Regression tests reject altered batch/non-batch strides and using the probe layout in generation. All 54 CPU/evidence tests and CLI checks passed [CI40262159433](https://github.com/amilcodes/RapidTinyRec/actions/runs/40262159433). No GPU rerun or measured source edit was made. Original failed stdout/stderr and frozen source are archived beside the separate corrected auditor snapshot and successful audit. Both auditor hashes are recorded.

[Published pilot evidence](../benchmarks/pilots/beam-replay) contains request and helper-probe CSVs, complete audit/summary, archive manifest and source/config/resource hashes. The archive has 281 files, including all 169 raw files, both source snapshots, original/revised audit logs, run/status records, independent review, exporter and its tests. Archive verification tests recompute every file hash, all submitted source hashes, raw request FP32 bits/repeats, latency strata and probe aggregates.

To reproduce the evidence audit, extract `evidence.tar.gz` into an empty directory. Set `EVIDENCE` to that directory, `HISTORICAL` to the pinned historical GQA run's `outputs/gqa_500_users`, and `TOKEN_MAP` to the pinned Office `added_tokens.json`. From the extracted corrected `audit-source` directory run:

```bash
python3 -m rapidtinyrec.audit_beam_replay "$EVIDENCE/raw/outputs" \
  --snapshot "$EVIDENCE/source-snapshot/rapidtinyrec" \
  --run-record "$EVIDENCE/run-record.json" \
  --historical "$HISTORICAL" --token-map "$TOKEN_MAP" \
  --output "$EVIDENCE/reproduced-audit.json"
```

The archive's `reproduce.json` and `status.json` retain the original absolute command/cwd and corrected command/cwd; relocation must be explicit. The source snapshot includes the exact OSCAR profile and `jobs/beam-replay.sh`. Hardware reproduction uses one allocated A6000 and fresh ABBA processes with the pinned release; no local/login-node inference. Existing pilot/full submission scripts have exclusive intent guards and must not be reused to duplicate these jobs.

The accepted folded full500 baseline remains **114.8412395/119.8975582 ms** (source `af2c05dae10a6b107616da964eb2a1360eb5cde1`, evidence `f21bd347f3e7dff2c91dac2967d506ed881eefce`). Main performance documentation remains unchanged pending full500. No FlashRec head-to-head or transferable published speedup is claimed.
