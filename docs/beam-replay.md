# Beam-state replay full500: exact scores, mixed latency result

On the same frozen 500 users, replaying the two original beam-state helpers preserves **all 25,000 ranks, 25,000 FP32 sequence scores and 125,000 transition scores** against both the paired folded control and historical compiled GQA. Maximum score deltas are zero, and every held-out SID ranking metric is unchanged.

Overall generation-call median improves **115.042567 → 114.209168 ms (0.724%)** and p95 **120.083272 → 119.355851 ms (0.606%)**. Both phase halves improve. However, length 111's median regresses **0.259660 ms (0.219%)**, in both halves, and length 116's p95 regresses **0.014144 ms (0.0118%)**. This misses the strict performance promotion gate by length. **Keep the accepted folded baseline and main performance documentation unchanged.** Retain this as an exact candidate with a modest aggregate benefit and mixed strata, not a uniformly faster replacement. No second full500 or adaptive routing around failed lengths is scheduled.

Full job **7954755**, run `beam-replay-full-20261026T030606Z-5721a23f41b7`, completed successfully in **21:12** on gpu1503: one NVIDIA RTX A6000, four CPUs, 32 GB, regular gpu/norm-gpu, 30-minute bound. Measured source [`9bd2a86b5b90594caea8a235b6c5ed56ada8a369`](https://github.com/amilcodes/RapidTinyRec/tree/9bd2a86b5b90594caea8a235b6c5ed56ada8a369) and all **49 submitted files/config** are byte-identical to the [pilot](beam-replay-pilot.md). That source passed 53 CPU/evidence tests and CLI checks in [CI40258692706](https://github.com/amilcodes/RapidTinyRec/actions/runs/40258692706). Full evidence preserves the hash-pinned pilot audit/review that authorized this run.

## What is measured

The isolated candidate invokes the unchanged bound HF 4.55.4 `_get_running_beams_for_next_iteration` and `_update_finished_beams`, with their original ATen arithmetic and 100/150-candidate top-k. It starts from accepted folded/profile PR12. Vocabulary-wide selection, both full-vocabulary normalizations, SID constraints, all five probabilities, original HF saved-score clones, termination synchronization and model/cache behavior remain unchanged. Precision remains FP32, TF32 off. Rejected packed-MLP and constraint replay behavior is not included.

Instance wrappers stage every dynamic input and keep independent outputs by helper, prompt length and step, within an explicit generation lease on one sequential CUDA stream. Capacity remains prompt+8 for five generated tokens. The generation pool has **40 model graphs plus 100 helper graphs** per candidate phase, all frozen before timing. Every final output and transition is consumed before lease close.

Fresh folded/beam/beam/folded processes form ABBA. The same 500 frozen users are divided into even/odd halves, with one timed request per user per variant. Compiler artifacts start cold in phases 0/1 and corresponding disk artifacts are reused in 2/3, with fresh model/GPU allocations and capture. ABBA pairs phases; it is not adjacent-request alternation or an independent-run variance study. Do not compound cross-run speedups.

Timing covers the synchronized generation call, including staging, wrappers and original saved-score clones. Input preparation, output/transition extraction, post-request diagnostic scans and file writes remain outside the timer. This is not total application request latency.

## Latency and regressions

| Stratum | Paired users | Folded median / p95 (ms) | Beam median / p95 (ms) |
|---|---:|---:|---:|
| Overall | 500 | 115.042567 / 120.083272 | 114.209168 / 119.355851 |
| Length 76 | 109 | 113.267574 / 115.133776 | 112.736437 / 113.883848 |
| Length 81 | 39 | 113.548119 / 115.206052 | 113.275338 / 114.376575 |
| Length 86 | 69 | 114.222560 / 115.608922 | 113.590680 / 114.793525 |
| Length 91 | 85 | 114.516392 / 116.279076 | 114.026058 / 114.926085 |
| Length 96 | 58 | 115.925966 / 117.435621 | 115.602467 / 116.537498 |
| Length 101 | 42 | 118.434979 / 120.097081 | 118.002858 / 118.759412 |
| Length 106 | 24 | 118.533036 / 120.476743 | 118.265023 / 118.940323 |
| Length 111 | 18 | 118.556682 / 119.991651 | 118.816342 / 119.272323 |
| Length 116 | 13 | 119.414434 / 119.925547 | 118.796531 / 119.939691 |
| Length 121 | 43 | 119.799923 / 121.029823 | 119.244419 / 120.332287 |
| Half 0 | 250 | 115.065783 / 120.254632 | 114.149524 / 119.401155 |
| Half 1 | 250 | 114.977304 / 119.574129 | 114.331491 / 119.255219 |

The candidate is faster in **333/500 individual pairs**; median paired-request saving is **0.408642 ms**. This differs from subtracting the two marginal medians. Length 111 median regressions are 0.254175 ms in half 0 (8 users) and 0.249764 ms in half 1 (10 users). Length 116 p95 is 0.243404 ms slower in half 0 (7 users) and improves in half 1 (6 users). Small tail strata are uncertain; their regressions remain visible rather than discarded. No statistical significance or run-to-run robustness is claimed.

All 20 length-by-half strata, individual timings and the conservative promotion decision are retained in the archived independent review and request CSV. The current same-allocation control is the primary comparison. The historical accepted folded baseline remains 114.8412395/119.8975582 ms, source `af2c05dae10a6b107616da964eb2a1360eb5cde1`, evidence `f21bd347f3e7dff2c91dac2967d506ed881eefce`.

## Exactness, stability and ranking quality

All **25,000/25,000 ranks, 25,000/25,000 sequence FP32 values and 125,000/125,000 transitions** match in the paired comparison. Both variants separately match historical GQA for all 500 outputs, with sequence/transition maximum deltas **0**. All 120 warmup outputs also match historical GQA; 40 warm repeated comparisons match 2,000 rank/sequence values and 10,000 transitions. Each user has one timed observation per variant, so there are no additional full500 timed-repeat samples. The pilot supplies separate six-trial evidence, not extra independent users here.

Seven GPU setup scenarios cover equal scores, ties crossing kth positions, signed zeros, near ties, infinities, finished flags and nonzero length penalty, over four mixed-length changed-input requests each. Across both candidate phases, each category—graph/direct, original direct-repeat and retained outputs after five steps—matches **3,248,000/3,248,000 values** with finite maximum delta zero. Integer sequence/ancestry and boolean values are checked along with FP32 bits. The 5,880 helper-output CSV comparisons and compressed raw direct/graph/repeated values are preserved.

Every candidate warm/timed request has **35 independent helper output storages**, **five independent original saved-score clones**, and an open lease during transition extraction. All 2,802,750 final sequence/score/ancestry values across the two phases match retained helper outputs. Original helper hashes, actual shapes/strides/dtypes/scalars, staged-byte counters, graph counts and disjoint storage pass. There is **zero measured compilation or capture**.

Held-out SID quality is identical for all 500 users in both variants. Every per-user rank and paired metric change is zero:

| k | Hit rate | NDCG | MRR | Paired changes |
|---:|---:|---:|---:|---|
| 1 | 0.058000 | 0.0580000000 | 0.0580000000 | All zero |
| 5 | 0.088000 | 0.0712985063 | 0.0660000000 | All zero |
| 10 | 0.114000 | 0.0795951243 | 0.0693650794 | All zero |
| 20 | 0.140000 | 0.0864149788 | 0.0713662263 | All zero |
| 50 | 0.200000 | 0.0980443836 | 0.0731133873 | All zero |

These are one held-out SID per unique user, without item-collision disambiguation. There is no observed ranking-quality loss on this fixed cohort. This exactness evidence is separate from latency uncertainty and does not establish behavior for every unseen workload.

## Memory, logical storage and setup

Allocator byte measurements are per fresh process. Inference peaks reset after setup. Logical storage and aggregate allocator totals are different quantities.

| Phase | Variant | Allocated | Reserved | Inference peak allocated | Inference peak reserved | Setup peak allocated | Setup peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Folded | 8,027,984,896 | 10,500,440,064 | 8,352,647,680 | 10,500,440,064 | 8,810,904,064 | 10,500,440,064 |
| 1 | Beam | 8,026,866,176 | 10,678,697,984 | 8,351,474,688 | 10,678,697,984 | 8,790,841,344 | 10,678,697,984 |
| 2 | Beam | 8,026,866,176 | 10,255,073,280 | 8,347,716,096 | 10,255,073,280 | 8,788,224,512 | 10,609,491,968 |
| 3 | Folded | 8,027,984,896 | 9,986,637,824 | 8,349,893,120 | 9,986,637,824 | 8,808,287,232 | 10,550,771,712 |

At the end of measurement, beam's aggregate allocated total is 1,118,720 bytes lower in each half, while reserved memory is 178,257,920 bytes higher in half 0 and 268,435,456 higher in half 1. This is not evidence that helper storage costs nothing: helper inputs total **11,117,550 logical bytes**, outputs **4,442,500**, and unique input/output storage **15,560,050 bytes**, excluding graph intermediates/private-pool slack. The aggregate measurements do not isolate the cause of the small allocated-total difference. Saved HF clones remain 152,265,000 bytes in both variants.

Staging per request ranges from 886,755 bytes at length 76 to 1,336,755 at 121. Including warmups, candidate phase staged totals are 285,423,850 and 284,173,850 bytes, independently checked against actual request lengths and signatures. Model logical/unique parameter storage is unchanged at **6,176,878,592 bytes**, including **3,082,813,440 gate/up bytes**.

Model graph inputs total **397,999,680 logical bytes** and **15,572,544 unique bytes**: shared prefix capacity 6,938,624, suffix capacity 8,601,600 and metadata 32,320. Prefix capacity is batch 1/121 tokens; suffix capacity batch 50/3 tokens. Dense exact-length attention K/V remains temporary per layer. Model replay output suffix storage totals 286,720,000 bytes. This candidate does not pack weights.

| Phase | Model load (s) | Combined setup (s) | Backend compilation (s) | Model capture/warmup (s) | Helper capture/warmup (s) |
|---|---:|---:|---:|---:|---:|
| 0 | 1.679704 | 357.169188 | 269.441045 | 217.465519 | 0.000000 |
| 1 | 1.662617 | 370.724610 | 273.524234 | 219.902763 | 0.710884 |
| 2 | 1.682193 | 166.154356 | 70.546996 | 86.973165 | 1.700313 |
| 3 | 1.700313 | 156.855199 | 69.380366 | 87.677656 | 0.000000 |

Combined setup includes probes/serialization, compilation, warmup and capture. These intervals nest/overlap and **must not be added**. Model loading is separate. Synthetic probes use separate setup pools; their graphs are not the 100 generation helper graphs.

## Audit provenance and reproduction

The pilot's original audit failed a structural assertion because its synthetic initial finished state slices/clones 50 of 100 candidates while preserving the singleton batch stride. Audit-only correction [`34b517a6a465f097f4d68e083eb1741cd9637009`](https://github.com/amilcodes/RapidTinyRec/commit/34b517a6a465f097f4d68e083eb1741cd9637009) validates the exact source-derived stride only for synthetic step 0; it retains strict generation/recurrent layouts and all numerical/index equality checks. Its 54 tests/CLI checks passed [CI40262159433](https://github.com/amilcodes/RapidTinyRec/actions/runs/40262159433). Original failed logs and measured source remain in the [pilot evidence](../benchmarks/pilots/beam-replay). No GPU rerun or measured code change was used to resolve that audit error.

The exact saved full500 audit runs from a separate immutable snapshot of this corrected auditor, while `--snapshot` points to original submitted source. It completed successfully on the first full500 audit execution. An independent standard-library review recomputes raw FP32 bytes, historical/warm/paired equality, probe hashes/indices, latency including every length-by-half stratum, and SID quality from the pinned token map. Its conservative performance decision remains false in the exported summary.

[Full500 evidence](../benchmarks/full500/beam-replay) includes all **1,193 raw files** within a **1,308-file archive**, both source snapshots, complete audit, run/resources/config records, independent review/script, exporter/tests, original submitter and full submitter, and hash-pinned pilot gate evidence. `requests.csv` contains 500 pairs; `helper-probes.csv` contains 5,880 comparisons. The archive manifest and `source.json` expose every hash. Publication tests verify all files, byte-identical pilot/full source, raw request bits, latency strata, preserved length regressions, audit provenance and probe aggregates. [Pilot report and evidence](beam-replay-pilot.md) remain separate.

Extract `evidence.tar.gz` into an empty directory. Set `EVIDENCE` to that directory, `HISTORICAL` to the pinned historical GQA run's `outputs/gqa_500_users`, and `TOKEN_MAP` to the pinned Office `added_tokens.json`. From extracted `audit-source` run:

```bash
python3 -m rapidtinyrec.audit_beam_replay "$EVIDENCE/raw/outputs" \
  --snapshot "$EVIDENCE/source-snapshot/rapidtinyrec" \
  --run-record "$EVIDENCE/run-record.json" \
  --historical "$HISTORICAL" --token-map "$TOKEN_MAP" \
  --output "$EVIDENCE/reproduced-audit.json"
```

`reproduce.json` retains the exact original absolute command/cwd; relocate paths explicitly. Submitted hardware reproduction is `jobs/beam-replay.sh --full-cohort` with archived `oscar.beam-replay.json` profile `beam-replay-full`, pinned release and one allocated A6000. The already-used submission scripts have exclusive intent guards: recover existing identities instead of duplicating jobs. No login-node/local inference.

Further work should move to a separately justified, bounded native FP32 backend diagnostic rather than expanding capture or tuning around these measured users. No lower precision, restricted-head scoring, omitted arithmetic or approximate pruning is authorized. FlashRec's native model/precision/score semantics remain unmatched; no head-to-head performance claim is made here.
