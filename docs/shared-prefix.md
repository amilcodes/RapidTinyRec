# Physical shared-prefix cache

The complete four-user A6000 pilot preserves all ranking and score bits while
reducing allocated memory by about 31.7% and median/p95 latency by 2.47%/2.41%.
The 500-user confirmation timed out during its final control phase. The candidate
finished all 500 and matches historical GQA exactly, but only 357 paired controls
were saved. The complete paired comparison remains unverified; both timeouts
are retained below.

## Completed pilot

Four users span the shortest/longest histories (76/121 tokens), with six paired
trials each. All 24 complete rank lists, 1,200 positions, 1,200 sequence values
and 6,000 transition values match bit for bit, with zero maximum deltas. Both
variants match historical compiled GQA exactly. Repeated requests are stable;
eight graphs per phase are frozen, with no measured compilation or capture.

| Metric | Shared inputs | Physical shared prefix |
| :--- | ---: | ---: |
| Median latency | 131.766 ms | 128.512 ms |
| p95 latency | 137.404 ms | 134.095 ms |
| Logical input-view bytes | 2.136 GiB | 0.074 GiB |
| Unique input storage | 0.331 GiB | 0.0145 GiB |
| Allocated GPU memory, phase range | 8.994–8.997 GiB | 6.142 GiB |
| Reserved GPU memory, phase range | 10.711–10.969 GiB | 7.418–7.660 GiB |
| Peak allocated during inference | 9.849–9.859 GiB | 6.436–6.442 GiB |
| Peak reserved during inference | 10.711–10.969 GiB | 7.418–7.660 GiB |

The physical prefix capacity is 6,938,624 bytes and suffix capacity 8,601,600
bytes; the remaining graph-input storage is token/position metadata. Prefix
batch size is one, suffix batch size is 50, and no prefix bytes are reordered.
Both history lengths and both phase halves improve at median and p95. These
are four-user pilot results, not evidence of a 500-user improvement. Memory
ranges retain the differences between cold and reused-artifact processes.

| Phase | Disk compiler cache | Setup/warmup | Backend compile, nested | Graph preparation, nested |
| :--- | :--- | ---: | ---: | ---: |
| 0: shared inputs | Cold | 134.261 s | 103.505 s | 0.768 s |
| 1: shared prefix | Cold | 335.686 s | 262.408 s | 205.128 s |
| 2: shared prefix | Reused from phase 1 | 143.238 s | 66.720 s | 80.097 s |
| 3: shared inputs | Reused from phase 0 | 62.926 s | 32.016 s | 0.766 s |

Graph preparation includes forward warmup, which triggers prefix compilation;
its interval overlaps the backend compilation total. Do not add these columns.
All setup is excluded from inference, including artifact reuse. Every process
still loads its model and creates fresh GPU allocations and CUDA graphs.
The full pilot completed in 12m52s using source `735a73b`.
[All paired samples, independent audit and immutable source/output hashes](../benchmarks/pilots/shared-prefix)
are published. ABBA pairs users across phases and does not establish cross-run
variance or concurrent-serving behavior.

## Full comparison: timed out

Job 7861959 (`prefix-full-20261023T190124Z-1ae3764cde99`) reached its
15-minute bound during phase 3. Phases 0–2 completed; phase 3 saved 107 of its
250 shared-input controls. The unchanged full audit correctly fails on the
missing phase-3 summary. These are **incomplete diagnostics**, not a completed
500-user ABBA performance result.

The shared-prefix candidate finished all 500 users. Against historical compiled
GQA, all 25,000 rank positions, 25,000 sequence values and 125,000 transition
values match bit for bit, with zero maximum deltas. Against the control from
this allocation, all 357 available rank lists, 17,850 positions/sequence values
and 89,250 transitions match exactly. The remaining 143 paired controls are
missing; historical latency is not substituted for them.

| Complete first half only: 250 paired users | Shared inputs | Shared prefix |
| :--- | ---: | ---: |
| Median latency | 129.600 ms | 126.838 ms |
| p95 latency | 137.361 ms | 133.363 ms |
| Allocated GPU memory | 18.783 GiB | 7.485 GiB |
| Reserved GPU memory | 22.668 GiB | 9.779 GiB |
| Peak allocated during inference | 19.639 GiB | 7.779 GiB |
| Logical input-view storage | 10.681 GiB | 0.371 GiB |
| Unique input storage | 0.331 GiB | 0.0145 GiB |

The first-half median/p95 reductions are 2.13%/2.91%. All ten length groups in
that half improve at both statistics. These isolated memory readings include
40 exact graphs per process; they are not interchangeable with the eight-graph
pilot. The second prefix phase allocated 7.485 GiB, reserved 9.313 GiB and peaked
at 7.784 GiB allocated. Phase 3 has setup memory only, not final measured peak
memory, so no complete four-phase memory range is reported. Prefix/suffix
capacity remains 6,938,624/8,601,600 bytes; unique input storage including graph
metadata is 15,572,544 bytes versus 355,565,120 for shared inputs.

| Phase | Disk artifacts | Setup/warmup | Backend compile, nested | Graph preparation, nested |
| :--- | :--- | ---: | ---: | ---: |
| 0: shared inputs | Cold | 137.173 s | 101.344 s | 3.787 s |
| 1: shared prefix | Cold | 346.237 s | 266.415 s | 211.000 s |
| 2: shared prefix | Reused | 147.296 s | 67.048 s | 82.279 s |
| 3: shared inputs | Reused | 67.915 s | 31.535 s | 3.936 s |

Setup/warmup consumed 698.621 seconds total; the saved inference calls consumed
111.021 seconds. Model loading, output extraction/serialization and other job
work also count toward the allocation. The scheduler stopped the process during
measurement; no preceding Python exception was found. All four phases froze 40
graphs before measurement. Completed phases record zero measured compilation
and capture; phase 3 has no final capture counters or memory peaks. Compiler
and graph preparation intervals overlap within setup and must not be added.

[Partial samples, exact comparisons and source/output hashes](../benchmarks/failures/shared-prefix-full-timeout)
are retained. No replacement was submitted. Finishing the comparison requires
a fresh, longer bounded allocation with the same source and four-phase method.

## Implementation and protocol

One request's history is stored once per layer, with batch size one. Each beam
stores only its generated suffix. Beam selection gathers these suffixes into
stable replay inputs, preserving duplicate parents without overwriting sources.
All exact-length graphs share the prefix and suffix input capacity.

The attention implementation is unchanged. Inside each layer, the shared prefix
is expanded and concatenated with the suffix into temporary contiguous dense
K/V tensors for native GQA. The cache retains only the single-row prefix and
beam suffix, not these temporary dense attention tensors. This does not remove
attention's dense history reads or implement a paged/shared-prefix attention
kernel. Graph-private intermediates can still consume memory; measure them.

The benchmark uses FP32 with TF32 disabled, the pinned Office checkpoint, the
same 500 users, beam 50, all three semantic-ID tokens plus newline/EOS scores,
and full-vocabulary normalization. No padding or score tolerance is introduced.

A bounded A6000 pilot precedes the full run. Four fresh processes execute in
ABBA order: shared inputs / prefix / prefix / shared inputs. The pilot uses four
users at the shortest/longest lengths and six trials per user; the full run
uses even and odd halves of the frozen cohort. Every phase warms every selected
length and freezes 8 pilot or 40 full graphs. Pairing is across phases, not
adjacent request alternation. One allocation cannot establish run-to-run variance.

Report logical/unique input bytes, prefix/suffix capacity, isolated CUDA allocated
and reserved memory and peaks, median/p95, exact rank positions and FP32 score
bits, plus maximum numerical deltas. Compile/capture/warmup are nested setup
measurements and excluded from inference. Source snapshots and raw outputs are
retained. The full run requires the pilot to pass exactness and ownership checks.

CPU tests cover the real constrained 50-beam HF loop with FP32 scores, multiple
history lengths, consecutive users, duplicate parents and arena ownership.
An initial CI check correctly rejected a change to the pinned prefill source;
the candidate now has a separate prefill wrapper, preserving that source/hash.
This failure and the corrected check remain in CI history.

## First pilot: incomplete

Job 7859244 timed out at 15m13s before phase 3 produced measured requests.
Phases 0–2 completed. The independent four-phase audit failed on the missing
phase-3 summary; this is not a passing pilot and does not authorize the full run.

The available first-half pairing covers four users, three trials each. All 600
rank positions and sequence values plus 3,000 transitions match bit for bit;
maximum score deltas are zero. Both completed prefix phases also match historical
GQA exactly. Partial median/p95 is 132.305/138.364 ms for shared inputs versus
128.270/133.832 ms for the prefix cache. Allocated memory is 8.994 versus 6.142 GiB.
These are incomplete first-half diagnostics, not the final ABBA comparison.
[Failure diagnostics and immutable output/source hashes](../benchmarks/failures/shared-prefix-timeout)
are retained.

Cold setup took 134.941 s for shared inputs and 341.915/339.368 s for the two
prefix processes. The prefix has separate compiler specializations for empty,
one-token and longer suffixes. Its backend compilation took 267.374/265.169 s.
Graph preparation (including forward warmup and its compilation) took
207.935/207.500 s. These intervals overlap. No compilation or capture occurred
in the completed measured phases.

## Bounded rerun

To fit four processes within 15 minutes, the second process for each variant
reuses its own on-disk compiler artifacts. Phases 0/1 start cold; phases 2/3
reuse artifacts from phases 1/0 respectively. Every process still loads the
model, allocates GPU memory and captures its own graphs from scratch. This
preserves isolated GPU memory measurements while avoiding duplicate code builds.
Report cold setup and artifact-reuse setup separately; do not call all four
phases cold starts. The model/cache/beam implementation is unchanged.

Within one allocated A6000 job, use a new output/cache root and run sequentially:

```bash
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 TORCH_LOGS=recompiles
for phase in 0 1 2 3; do
  case "$phase" in 0|3) variant=arena;; 1|2) variant=prefix;; esac
  export TORCHINDUCTOR_CACHE_DIR="$PWD/results/prefix-pilot/cache/$variant/inductor"
  export TRITON_CACHE_DIR="$PWD/results/prefix-pilot/cache/$variant/triton"
  extra=()
  if (( phase >= 2 )); then extra+=(--reuse-compiler-artifacts); fi
  python -m rapidtinyrec.benchmark_shared_prefix \
    --release-root releases/office --reference-results results/gqa/gqa_500_users \
    --output-root results/prefix-pilot --phase "$phase" "${extra[@]}" || exit 1
done
python -m rapidtinyrec.audit_shared_prefix results/prefix-pilot \
  --historical results/gqa/gqa_500_users \
  --snapshot frozen-source/rapidtinyrec --run-record run-record.json \
  --output results/prefix-pilot/audit.json
```

After a successful pilot, use new output/cache directories and add
`--full-cohort` to each phase. Sequential single-request inference only;
concurrent requests on the same arena are unsupported.
