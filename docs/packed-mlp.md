# Decode MLP gate/up packing — rejected by FP32 exactness

The completed four-user pilot preserves every ranking but changes sequence and transition score bits. **It does not qualify for the 500-user comparison.** Paired median/p95 decreased from **115.690/119.731 ms to 111.329/115.296 ms**, but exact score equality is required. These are pilot measurements from one allocation, not a validated full500 improvement.

Job `7907979`, run `packed-mlp-pilot-20261024T215118Z-533138a5b479`, completed four phases on one NVIDIA RTX A6000 (`gpu1503`) in 18m19s, exit `0:0`. Measured source: `9b94664b1800260674d63a26cdc33eb5a4cb5370`. All 56 submitted files match the frozen snapshot and earlier cancelled pilot. The independent auditor passed integrity checks, saved every comparison, and exited **1** because FP32 equality failed. No full500 job was submitted.

Evidence: [summary](../benchmarks/pilots/packed-mlp/summary.json), [24 paired requests](../benchmarks/pilots/packed-mlp/requests.csv), [complete audit](../benchmarks/pilots/packed-mlp/audit.json), [source and evidence hashes](../benchmarks/pilots/packed-mlp/source.json), [raw archive](../benchmarks/pilots/packed-mlp/evidence.tar.gz), and [archive file manifest](../benchmarks/pilots/packed-mlp/archive-manifest.json).

## Exactness and gate decision

| Timed comparison | Rank positions matching | Sequence FP32 values matching | Transition FP32 values matching | Max sequence delta | Max transition delta |
|---|---:|---:|---:|---:|---:|
| Packed vs paired folded | 1,200 / 1,200 | 78 / 1,200 | 1,524 / 6,000 | 0.000030517578125 | 0.000019073486328125 |
| Packed vs historical GQA | 1,200 / 1,200 | 78 / 1,200 | 1,524 / 6,000 | 0.000030517578125 | 0.000019073486328125 |
| Folded vs historical GQA | 1,200 / 1,200 | 1,200 / 1,200 | 6,000 / 6,000 | 0 | 0 |

All 24 paired rank lists match. None of the 24 packed sequence vectors or transition arrays is wholly exact: **1,122 sequence values and 4,476 transitions differ**. The discrepancy occurs in both halves. All six timed repeats per user and variant are bit stable across fresh processes.

Every saved warmup was checked. The folded phases' 24 warmup outputs match historical GQA in all 1,200 ranks, 1,200 sequence values and 6,000 transitions. The packed phases' 24 warmup outputs match all 1,200 ranks, but only 78 sequence values and 1,524 transitions, with the same maximum deltas as timed requests. Within each phase, the four dense-forward/replay pairs and replay repeats match exactly. Matching the candidate's own warmup reference does not establish historical equality.

Changing decode GEMM dimensions preserves installed weight bits but is not output-bit equivalent on this workload. These results do not isolate a particular low-level accumulation or kernel-selection cause. CPU tolerance and mathematical equivalence do not override the GPU gate.

## Request latency

Four frozen users cover lengths 76 and 121, with six paired trials each. Values below are end-to-end inference milliseconds, including splitting packed outputs and all request work. P95 uses inclusive interpolation.

| Scope | Pairs | Folded median | Packed median | Folded p95 | Packed p95 |
|---|---:|---:|---:|---:|---:|
| All pilot requests | 24 | 115.690017 | 111.328881 | 119.731143 | 115.295713 |
| 76 tokens | 12 | 112.751013 | 108.307516 | 113.353589 | 109.012906 |
| 121 tokens | 12 | 119.282855 | 114.626462 | 119.938563 | 115.588143 |
| Half 0: repeats 0–2, phases 0/1 | 12 | 115.560319 | 111.240366 | 119.279124 | 114.743025 |
| Half 1: repeats 3–5, phases 3/2 | 12 | 116.210045 | 111.328881 | 119.938563 | 115.588143 |

Pilot median decreases 3.770% and p95 3.704%; both lengths and halves improve. The numerical failure still blocks promotion. ABBA pairs the same user/repeat across separate phases; this is neither adjacent request alternation nor a run-to-run variance study. No cross-run speedups are compounded.

The validated folded-query full500 result remains **114.8412395 ms median / 119.8975582 ms p95**, source `af2c05dae10a6b107616da964eb2a1360eb5cde1`, evidence `f21bd347f3e7dff2c91dac2967d506ed881eefce`. This pilot's paired folded control is the primary comparison. Four users cannot replace the 500-user distribution.

## Setup, model loading, and packing

All times are seconds. Every phase reloads the model and makes fresh GPU allocations and graphs. Phases 0/1 use cold variant-specific disk compiler artifacts; 2/3 reuse the corresponding variant's artifacts.

| Phase / variant | Artifacts | Model load | Setup after load | Packing in setup | Backend compilation in setup | Graph capture/warmup in setup |
|---|---|---:|---:|---:|---:|---:|
| 0 / folded | Cold | 1.668886 | 356.058587 | 0 | 269.730013 | 218.358851 |
| 1 / packed | Cold | 1.678366 | 355.359622 | 0.058144 | 267.128098 | 214.208604 |
| 2 / packed | Reused | 1.665330 | 150.550736 | 0.054663 | 69.107163 | 83.817099 |
| 3 / folded | Reused | 1.707664 | 145.977341 | 0 | 67.643170 | 81.577896 |

Setup includes packing, compiler work, graph preparation and warmup. These intervals nest and overlap and **must not be added**. Model loading precedes setup; none is included in inference. Reused-artifact setup is 4.573 s longer for packed. All phases retain eight warmed, frozen graphs with unchanged capture events, zero measured backend compilations, and zero measured captures.

## Memory and ownership

Values are GiB (2^30 bytes), rounded to six decimals; exact byte counts are in the audit. Setup and inference peaks cover separate windows.

| Phase / variant | Inference allocated | Inference reserved | Inference peak allocated | Inference peak reserved | Setup peak allocated | Setup peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| 0 / folded | 6.142110 | 7.417969 | 6.436465 | 7.417969 | 7.833670 | 8.585938 |
| 1 / packed | 6.161443 | 8.531250 | 6.456568 | 8.531250 | 7.862616 | 10.625000 |
| 2 / packed | 6.169454 | 8.531250 | 6.464579 | 8.531250 | 7.034433 | 9.021484 |
| 3 / folded | 6.142110 | 7.660156 | 6.441910 | 7.660156 | 7.012719 | 7.660156 |

Packing retains no second gate/up copy: 28 layers use disjoint contiguous row views in one allocation, preserving **3,082,813,440 unique gate/up bytes** and **zero persistent extra weight bytes**. Ownership is checked at installation, after warmup and after measurement. Original weight bits are preserved at installation. Additional allocator usage and larger setup peaks remain visible above; zero duplicate weights does not mean zero memory overhead.

Each phase retains **79,599,936 logical graph-input bytes** backed by **15,546,688 unique bytes**: prefix capacity **6,938,624**, suffix capacity **8,601,600**, metadata **6,464** bytes. Physical prefix batch is 1 with capacity 121 tokens; suffix batch is 50 with capacity 3 tokens. Attention still materializes temporary dense exact-length K/V. Prefix bytes reordered remain zero. These eight-graph pilot capacities differ from the forty-graph full500 baseline.

## Candidate and fixed protocol

The [folded-query profile](folded-profile.md) attributed 27.788–27.933 ms of diagnostic decode GPU activity to separate gate/up projections. This candidate replaces two `1536→8960` FP32 decode multiplications with one `1536→17920` multiplication and gate/up views. Prefill retains separate projections; SiLU, product and down projection are unchanged. The diagnostic profile motivated this test; it was not an end-to-end latency result.

Both variants retain folded-query attention, physical-prefix caching, FP32/TF32-off, the pinned Office checkpoint, the frozen 500-user source cohort, beam 50, all five probabilities and both full-vocabulary normalizations. Only four selected users are measured here. Process order is folded / packed / packed / folded. Resources: one regular `gpu` / `norm-gpu` A6000, four CPUs, 32 GB, 30-minute limit. All 46 original CPU/evidence tests and CLI checks passed [source CI 40100234692](https://github.com/amilcodes/RapidTinyRec/actions/runs/40100234692); that was not GPU bit-equality proof.

## Evidence and reproduction

The archive contains every completed pilot raw output, generated graph, log and manifest; all 56 frozen submitted files plus `.submission-config.json`; the run record; the exact audit invocation and failure stderr; the exporter; and four unchanged historical GQA reference output files. Its 228 files are individually hashed in `archive-manifest.json`; the archive is hashed in `source.json`. CSV fingerprints use compact JSON rankings and big-endian FP32 score bytes. Raw pilot manifests include the four selected dataset inputs. The full historical 500-user manifest stays at the original audit path; its file hash is recorded in the archive.

Complete unmodified local results and source snapshots also remain under `.oscar/results/RUN` and `.oscar/source_snapshots/RUN` in the replay harness. Original historical GQA outputs and manifest remain in the historical harness. For the full audit, use that original historical result directory as `GQA_RESULTS`:

```bash
mkdir -p EVIDENCE_DIR
tar -xzf benchmarks/pilots/packed-mlp/evidence.tar.gz -C EVIDENCE_DIR
python -m rapidtinyrec.audit_packed_mlp EVIDENCE_DIR/raw/outputs \
  --historical GQA_RESULTS --snapshot EVIDENCE_DIR/snapshot/rapidtinyrec \
  --run-record EVIDENCE_DIR/provenance/run-record.json --output AUDIT.json
```

Expected: integrity passes, `exactness_passed` is false, comparisons are written, exit status is 1. The exact original absolute command/cwd is in `provenance/audit-command.json`. CPU evidence checks recompute paired and historical score comparisons, warmup equality, repeat stability and latency from archived outputs, verify archive/source hashes, and retain the rejected result.

The frozen `snapshot/jobs/packed-mlp.sh` and `snapshot/oscar.packed-mlp.json` retain exact launch/resources. Reproducing GPU measurements requires a fresh authorized allocation, output root and variant-specific compiler-cache directories, with phases 0–3 in fresh processes:

```bash
python -m rapidtinyrec.benchmark_packed_mlp \
  --release-root RELEASE --reference-results GQA_RESULTS \
  --output-root OUTPUT --phase PHASE
```

Phases 2/3 reuse the same variant's earlier disk artifacts and add `--reuse-compiler-artifacts`; models, allocations and captures remain fresh. The pilot is the default. This result does **not** authorize `--full-cohort` or another candidate.

## Authorization and historical unrun pilot

The user authorized resubmission after the six-hour window closed. Job `7907979` was submitted October 24, 2026 at 21:51 UTC (17:51 America/New_York), retained through the queue, and completed successfully. The old cutoff did not apply. The renewed scope ends with this audited negative result; full500 is blocked by exactness.

Earlier job `7893602`, run `packed-mlp-pilot-20261024T135738Z-d85533e2a0f0`, was cancelled while PENDING at the previous deadline with zero compute time. Its empty outputs and missing-output audit remain historical records. See the unchanged [unrun record](../benchmarks/unrun/packed-mlp/summary.json), [source hashes](../benchmarks/unrun/packed-mlp/source.json), and [six-hour closeout report](research-window-2026-10-24.md).
