# Constraint replay: exact, no useful latency improvement

The A6000 pilot preserved every checked output bit but did not improve generation-call latency consistently. Median was **116.653839 ms versus 116.611049 ms** for the paired folded control (0.0367% slower); p95 was **119.815506 ms versus 120.723462 ms** (0.7521% lower). Length 76 median regressed 0.403%. Persistent allocated memory increased **310,564,352 bytes**. **No full500 run is authorized by this result; this candidate is not promoted.**

Measured source: `899ce4ecb05146910a05ce3a007ce2c621cefbe7`.
Run: `constraint-replay-pilot-20261026T011855Z-4b7e1f8d3254`, OSCAR job `7952157`, completed with exit 0 on `gpu1503` in 18:37. One regular A6000, gpu/norm-gpu, four CPUs, 32 GB host memory, 30-minute bound. All 48 submitted files match their frozen hashes. Source CI passed 50 CPU/evidence tests and CLI checks in [40253497914](https://github.com/amilcodes/RapidTinyRec/actions/runs/40253497914).

The primary comparison is the paired control in this allocation. The accepted folded-query full500 baseline remains 114.8412395/119.8975582 ms median/p95, measured source `af2c05dae10a6b107616da964eb2a1360eb5cde1`, evidence `f21bd347f3e7dff2c91dac2967d506ed881eefce`. Cross-run speedups are not compounded.

## Protocol and timing

Four fresh processes ran folded/constraint/constraint/folded ABBA with four frozen users at prompt lengths 76/121 and six paired trials (24 paired requests). Phases 0/1 had cold compiler artifacts; phases 2/3 reused the corresponding variant's disk artifacts with fresh model allocations and graph captures. Both variants use validated folded-query attention and physical-prefix caching, FP32 with TF32 off, the pinned Office checkpoint, beam 50 and vocabulary 152,265. Both full-vocabulary normalizations, all five scored probabilities, original ATen arithmetic/shapes and HF beam top-k/tie behavior remain unchanged.

Generation timing includes score/token staging, output splitting where applicable, original HF saved-score clones, and all work inside the synchronized `generate` call. It excludes input preparation, output/transition extraction, post-request audit scans and file writes. This is not total serving latency. Eight model graphs per phase and five additional candidate constraint graphs were warmed and frozen; no measured compilation or capture occurred. ABBA pairs phases; it is not adjacent request alternation or a run-to-run variance study. Small differences here do not establish statistical significance.

## Latency

All values are milliseconds. Half 0 is repeats 0–2; half 1 is repeats 3–5. Each stratum reuses the same users.

| Stratum | Pairs | Folded median | Constraint median | Folded p95 | Constraint p95 |
|---|---:|---:|---:|---:|---:|
| Overall | 24 | 116.611049 | 116.653839 | 120.723462 | 119.815506 |
| Length 76 | 12 | 112.870736 | 113.325205 | 114.651398 | 114.438855 |
| Length 121 | 12 | 119.689235 | 119.257780 | 120.918854 | 119.862057 |
| Half 0 | 12 | 117.013633 | 116.834089 | 120.918854 | 119.862057 |
| Half 1 | 12 | 116.117330 | 116.069414 | 119.960804 | 119.608550 |
| 76/half0 | 6 | 113.489216 | 113.302346 | 114.785388 | 114.663331 |
| 76/half1 | 6 | 112.870736 | 113.334954 | 113.806711 | 113.675157 |
| 121/half0 | 6 | 120.042481 | 119.460128 | 121.025356 | 119.881192 |
| 121/half1 | 6 | 119.235426 | 119.016038 | 120.020345 | 119.666269 |

## Fidelity, lifetime and quality

All **1,200/1,200 paired rank positions**, **1,200/1,200 sequence FP32 values** and **6,000/6,000 transition FP32 values** match bit for bit. Maximum sequence and transition deltas are both zero. Both variants also match historical compiled GQA for every timed request (48 total) and every saved warm output (48 total). Repeated-request comparisons pass: 20 comparisons per variant, plus 16 warm-repeat comparisons, with zero score deltas.

Each candidate phase passes 20 graph/direct and 20 retained-output comparisons over valid, unknown, mixed and repeated prefixes. Each category checks 152,265,000 FP32 values per phase. Across candidate phases, 1,522,650,000 saved HF score-clone values match graph outputs bit for bit after generation. Five independent saved-score storages remain disjoint from five independent graph outputs. Unknown-prefix fallback, mixed lengths, lease lifetime, staging counts and frozen storage remain valid; transition extraction precedes lease release.

Held-out SID quality is identical for the four unique users. Target ranks are miss, 1, 45 and miss. Hit rate/NDCG/MRR at 1, 5, 10 and 20 are each 0.25 in both variants. At 50, hit rate is 0.5, NDCG is 0.2952606492 and MRR is 0.2555555556. Every paired per-user metric change is zero. Six repeats are not six independent user samples. This four-user pilot establishes no population quality-equivalence claim; these are SID metrics without item-collision disambiguation.

## Memory and storage

Bytes below are per isolated process. Inference peaks and setup peaks cover different intervals; allocator reservations are not unique tensor bytes.

| Phase / variant | Allocated | Reserved | Inference peak allocated | Inference peak reserved | Setup peak allocated | Setup peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| 0 / folded | 6,595,040,256 | 7,964,983,296 | 6,911,101,440 | 7,964,983,296 | 8,411,338,752 | 9,219,080,192 |
| 1 / constraint | 6,905,604,608 | 8,841,592,832 | 7,227,512,832 | 8,841,592,832 | 8,723,541,504 | 9,307,160,576 |
| 2 / constraint | 6,905,604,608 | 8,841,592,832 | 7,227,512,832 | 8,841,592,832 | 7,840,794,112 | 8,841,592,832 |
| 3 / folded | 6,595,040,256 | 8,225,030,144 | 6,916,948,480 | 8,225,030,144 | 7,529,849,856 | 8,225,030,144 |

Both variants retain 6,176,878,592 logical and unique model-weight bytes, including 3,082,813,440 unique gate/up bytes. Model graph inputs total 79,599,936 logical bytes and 15,546,688 unique storage bytes. Physical-prefix capacity is 6,938,624 bytes (batch one, up to 121 tokens); suffix capacity is 8,601,600 bytes (batch 50, up to three tokens), plus 6,464 metadata bytes. Prefix storage is not reordered; temporary dense exact-length attention materialization remains unchanged.

Constraint staging inputs occupy 152,271,000 logical bytes and outputs 152,265,000 bytes; their unique combined storage is 304,536,000 bytes. Graph intermediates and private pools are additional. The original five HF saved-score clones occupy 152,265,000 bytes in both variants. Candidate allocated residency increases by 310,564,352 bytes. Each candidate measurement half stages 1,827,252,000 bytes across 12 requests and 60 replays; these copies remain in generation timing.

## Setup

Seconds below are separately observed intervals. **Setup contains compilation, preparation probes and graph warmup/capture; those intervals nest or overlap and must not be added.** Model loading is outside setup. No MLP packing is performed.

| Phase | Model load | Setup | Backend compilation | Model graph warmup/capture | Constraint graph warmup/capture |
|---|---:|---:|---:|---:|---:|
| 0 folded | 1.743189 | 351.794229 | 269.374064 | 213.084115 | — |
| 1 constraint | 1.717094 | 357.820793 | 270.702612 | 219.260317 | 0.036205 |
| 2 constraint | 1.693769 | 152.691896 | 69.957440 | 85.572917 | 0.036334 |
| 3 folded | 1.697866 | 151.185829 | 67.732304 | 85.967081 | — |

## Implementation and scope

Existing folded-profile traces at lengths 76/101/121 show five processor calls
on contiguous `[50,152265]` FP32 scores. A score matrix occupies 30,453,000 bytes.
The prototype keeps five separate score inputs and five outputs, plus 6,000 bytes
of staged token suffixes: 304,536,000 logical input/output bytes. Captured masks,
normalization/lookup intermediates and allocator pool slack are additional and
must be measured. Existing prefix/suffix model storage remains unchanged.

Each request obtains a processor lease from `ConstraintReplayPool.processor()`.
It stages the scores and last three integer token IDs, then replays the graph
for that step. Five independent graph-private pools keep processor outputs
alive through subsequent steps. The pinned [HF 4.55.4 beam search](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/generation/utils.py#L4105)
clones these outputs for `output_scores`, before adding cumulative beam scores.
Those existing timed clones stay unchanged and add 152,265,000 bytes of live
saved-score storage at five steps in both variants. Outside generation timing,
the audit checks the clones remain bit-identical to the independent graph outputs. The caller must consume transition-score and
other saved-score outputs before calling `lease.close()`; overlapping request
leases and use after close are rejected. The pool supports one sequential CUDA
stream, one score shape per step, and precisely the five validated steps.

Preparation warms and captures the same ATen operations without `torch.compile`.
`freeze()` requires all five graphs and no live lease. Capturing after freeze,
changing score layout/dtype/device, and using another stream are errors. Both
score copies and integer suffix copies remain inside generation-call timing.
Buffer reuse is an explicit borrowing contract, not general concurrent serving.


This isolated branch is based on #12. Rejected wide/batched packed-MLP evidence remains in #13. Graph-resident constraint work was inspired by FlashRec's orchestration, but retains this workload's FP32, full-vocabulary scoring contract. No head-to-head FlashRec benchmark or native-system speedup is claimed.

## Evidence and reproduction

[Published evidence](../benchmarks/pilots/constraint-replay/) contains the full audit, paired request CSV, summary, provenance and a hash manifest. `evidence.tar.gz` retains all 167 fetched raw files (including logs and generated graphs), 48 submitted source files plus the submission configuration, run record, original status, exact audit command, exporter, submission guard, exporter regression tests and independent review: 228 files total. Raw and frozen snapshot bytes are preserved. Publication tests independently verify every archive hash, reconstruct paired raw bit values and repeats, recompute median/p95 by length/half, and ensure the negative performance gate is retained.

To verify the published bundle without a GPU:

```bash
python3 -m unittest discover -s tests -p test_constraint_replay_evidence.py -v
```

Extract the archive into a new directory. Run the auditor from its frozen source snapshot, relocating paths explicitly to the extracted raw files and archived run record. Supply the pinned historical GQA results and Office token map used by the original run (their identities are checked):

```bash
python3 -m rapidtinyrec.audit_constraint_replay ../raw/outputs \
  --historical /absolute/path/to/gqa_500_users \
  --snapshot rapidtinyrec --run-record ../run-record.json \
  --token-map /absolute/path/to/Office_ckpt/added_tokens.json \
  --output /absolute/path/to/audit.json
```

`reproduce.json` preserves the original exact audit command/cwd and resources. `source-snapshot/jobs/constraint-replay.sh` and the frozen configuration preserve the four-phase GPU protocol. The existing job must not be resubmitted. This negative result does not permit a full500 run. Further experiments require a separately justified candidate based on the validated folded control.
