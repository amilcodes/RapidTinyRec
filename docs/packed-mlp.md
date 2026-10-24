# Decode MLP gate/up packing — candidate awaiting GPU measurement

The [folded-query profile](folded-profile.md) attributes 27.788–27.933 ms of diagnostic decode GPU activity to the separate MLP gate/up projections. The generated Inductor source shows two adjacent `1536→8960` FP32 matrix multiplications on the same input. This experiment tests one `1536→17920` multiplication followed by gate/up views, leaving SiLU, multiplication, and the down projection unchanged. Shape attribution comes from diagnostic traces; actual serving replay remains the primary trace, and no kernel timing establishes an end-to-end improvement.

`packed_mlp.py` changes only three-dimensional single-token decode inputs. Prefill keeps its separate projections. Weight packing occurs once after model loading: original gate/up parameters become disjoint contiguous row views of the packed allocation. The 28 layers retain 3,082,813,440 unique gate/up bytes, with zero persistent duplicate weight bytes. Setup peaks still include transient packing storage. Ownership is checked at installation, after warmup, and after measurement. Installation verifies every original weight bit; this does not prove output equality after changing GEMM dimensions.

Both variants use the validated folded-query attention and physical-prefix cache. FP32, TF32 off, Office checkpoint, beam50, all five probabilities, and both full-vocabulary normalizations remain fixed. The four fresh ABBA processes compare folded/packed on the first half, then packed/folded on the second. Phases 0/1 have cold disk compiler caches; 2/3 reuse the matching variant's artifacts but reload models, allocate GPU storage, and capture graphs independently.

The bounded pilot uses four frozen users at lengths 76/121 and six paired trials, with eight frozen graphs per process. A useful latency improvement and exact rankings, sequence scores, transition scores, repeat stability, and ownership are required before the frozen 500-user comparison (40 graphs per phase). Every saved output is independently compared with historical compiled GQA. Numerical failures remain evidence and do not advance to 500 users.

Setup begins after model loading and includes weight packing, compiler work, graph preparation, and warmup. Model loading and packing are also recorded separately. Compiler/capture intervals overlap within setup and must not be added. Measured inference includes all request work, including splitting packed outputs. Median/p95, both halves and each length, memory allocation/reservation/peaks, and prefix/suffix input storage will be reported after audit. Pairing is across ABBA phases, not adjacent request alternation or a study of run-to-run variance.

The independent auditor is `python -m rapidtinyrec.audit_packed_mlp`; it saves comparisons and explicitly returns nonzero for any output bit mismatch. CPU math tests allow tolerance only for functional validation; they do not establish GPU score equality. No GPU result or speedup is claimed yet.


## Window closeout: pilot never started

Job `7893602`, run `packed-mlp-pilot-20261024T135738Z-d85533e2a0f0`, remained queued until the six-hour research deadline on October 24. It was cancelled while PENDING, with zero compute time. No GPU latency, score equality, setup, or memory result exists, and no full500 run was submitted. This is an unrun candidate; all 46 CPU/evidence checks passed at source `9b94664b1800260674d63a26cdc33eb5a4cb5370`.

All 56 source files were reverified against the immutable snapshot and remote submission manifest. The saved measurement audit failed because phase outputs do not exist. See the [unrun record](../benchmarks/unrun/packed-mlp/summary.json), [source hashes](../benchmarks/unrun/packed-mlp/source.json), and [research-window report](research-window-2026-10-24.md). Further GPU work needs a new authorized window and a passing pilot before any full-cohort run.

For reproduction in an authorized allocation, use a fresh output root and variant-specific compiler-cache directories. Run phases 0–3 in fresh processes; phases 2/3 reuse the matching earlier disk artifacts and pass `--reuse-compiler-artifacts`:

```bash
python -m rapidtinyrec.benchmark_packed_mlp \
  --release-root RELEASE --reference-results GQA_RESULTS \
  --output-root OUTPUT --phase PHASE
python -m rapidtinyrec.audit_packed_mlp OUTPUT \
  --historical GQA_RESULTS --snapshot SNAPSHOT/rapidtinyrec \
  --run-record RUN_RECORD --output AUDIT.json
```

The pilot is the default. `--full-cohort` is gated on complete pilot exactness and a useful latency improvement. The experiment must retain FP32/TF32-off, the pinned release and all probability normalization; setting a larger GEMM is not permission to relax score bits.
