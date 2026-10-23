# Decode replay pilot

This branch tests whether CUDA graph replay reduces the launch cost of the
compiled native-GQA decoder. Staged replay produced almost no aggregate gain.
Direct beam gathers into graph input buffers improved the four-user pilot by
3.77% at median latency. The full 500-user confirmation below preserves every
rank and score bit, with a 3.96% median reduction.

The first A6000 pilot stopped during warmup, before capture or measured requests:
Hugging Face passed `inputs_embeds=None` and the argument validator rejected the
keyword. The validator now preserves that placeholder while still rejecting
actual embedding tensors. A CPU integration test runs the real constrained
50-beam generation loop through staging and compares all five scores. This fixes
the observed compatibility error. The corrected A6000 rerun completed and passed
its independent audit; the failed attempt remains part of the experiment record.

## Pilot result

| Four users, eight paired repeats | Median, ms | p95, ms |
| :--- | ---: | ---: |
| Compiled GQA | 138.145 | 145.429 |
| CUDA replay with staged KV | 137.728 | 144.577 |

The aggregate change is only 0.30% median / 0.59% p95. For 76-token histories,
median moved 132.845 → 131.006 ms; for 121-token histories it regressed
143.061 → 143.765 ms. Four selected users cannot establish a full-cohort speedup.

All 32 full rank lists, 1,600 ranked positions, 1,600 sequence scores and 8,000
transition scores match bit for bit, with zero numerical deltas. Repeated outputs
are stable, and the compiled control matches its historical outputs exactly.

Eight captured shapes require 2.14 GiB of input buffers. Total allocated/reserved
CUDA memory at completion was 10.80 / 12.89 GiB. Setup and warmup took 138.576 s;
the nested backend compilation total was 107.984 s and graph preparation was
0.765 s. No compilation or capture occurred during measured requests. These costs
are excluded from the latency table and must not be added together.

[Pilot fingerprints and independent audit](../benchmarks/pilots/decode-replay)
retain all samples. A second allocation reproduced the small gain: compiled GQA
137.016 / 143.835 ms median/p95 versus replay 135.864 / 142.631 ms (0.84% / 0.84%).
All 1,600 positions and sequence values plus 8,000 transitions remain exact.
The 500-user README performance table is unchanged.

## What profiling found

Four traces and twelve light profiles were recorded after that second benchmark.
All 800 instrumented positions/sequence values and 4,000 transitions match the
uninstrumented outputs bit for bit. No compilation or capture occurred.

| Actual CUDA activities per request | 76-token history | 121-token history |
| :--- | ---: | ---: |
| Compiled decode kernels, four forwards | 82.94 ms | 88.57 ms |
| Replayed decode kernels, four forwards | 84.31 ms | 87.81 ms |
| Additional replay staging copies | 3.19 ms | 4.80 ms |
| Beam reorder kernels in replay | 3.68 ms | 5.38 ms |

Replay executes the same 2,360 decode kernels. It reduces host time spent inside
those forward calls from about 43 ms to roughly 8 ms of staging, validation,
replay and cache-container calls, but most dispatch already overlaps GPU work.
These host and device times must not be added. Staging adds 236 device copies per
request, including 224 KV tensors. Matrix multiplication dominates the compiled
control's decode kernels: approximately 62.5–63.4 ms, including vocabulary
projection. Attention accounts for another 15.9–18.8 ms.

Direct beam reordering (below) writes KV values into the next graph’s stable
input buffers. This removes the extra staging traffic;
it does not remove the model's matrix work. Trace timings are diagnostic and
include profiler effects; the latency figures above come from separate ordinary
requests. [Profile audit and second pilot samples](../benchmarks/pilots/decode-replay-profile)
include every attributed kernel/copy and trace hashes.

Each exact decode shape owns stable input and KV tensors. Incoming cache values
are copied into those tensors before replay; that staging cost stays in request
latency. Cache sequence lengths are unchanged. Captured output tensors are
returned through fresh DynamicCache containers so beam reordering cannot replace
the persistent tensor references. Prefill still uses the compiled forward.

The first pilot uses two users from each end of the frozen history-length range,
with eight paired repetitions. It caps the pool at eight graphs and records input
buffer bytes, capture time and CUDA allocation/reservation. All 50 rankings and
all five transition scores are retained; replay/control comparisons and repeated
request stability checks run outside timing. These are pilot distributions, not
500-user median/p95 estimates.

```bash
python -m rapidtinyrec.run decode-pilot --release-root releases/office \
  --reference-results results/gqa/gqa_500_users --output-root results/replay-pilot
python -m rapidtinyrec.audit_decode_pilot results/replay-pilot/decode_replay_pilot \
  --historical results/gqa/gqa_500_users \
  --snapshot frozen-source/rapidtinyrec --run-record run-record.json \
  --output results/replay-pilot/audit.json
```

The standalone audit verifies the immutable source hashes in the run record,
the pinned cohort and four-user selection, repeated request stability, every FP32
score bit, all five summed terms, historical control outputs and timing summaries.
It also checks that no compilation or graph capture occurred after warmup. Its
integrity verdict is separate from the reported output-equality counts. The run
record lists `files` and their `sha256` digests relative to `frozen-source/`, with
the allocation's GPU count/type under `resources`.

For the next diagnostic, add `--profile` to the pilot command and use a fresh
output directory. All ordinary paired trials finish before instrumentation.
Three light profiles and one CPU/CUDA trace per variant and history length then
separate compiled prefill/decode, input staging, graph replay, cache expansion,
beam reordering and generation bookkeeping. Instrumented output bits must match
the earlier requests. CUDA event spans include stream idle time; actual GPU work
must be read from trace kernel activities. Traces are diagnostic, not latency
benchmark samples.

Run on one A6000 with the pinned environment. No graph capture or compilation is
permitted after warmup. A capture failure is a failed experiment; the implementation
does not silently fall back to eager execution. CPU tests cover buffer ownership,
shape rejection, beam reordering and tiny Qwen2 forward equality after duplicate
parent selection. These are CPU checks; a successful CUDA pilot is required
before a full-cohort run or any performance claim.

The pool supports sequential inference with one model and stream. Returned graph
tensors are borrowed until that shape is replayed again. It does not implement
concurrent serving, shared-prefix storage or variable-length graph padding.

The capture/warmup and persistent-tensor protocol follows the
[PyTorch CUDA graph guidance](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

## Direct-reorder candidate

`--direct-reorder --repeats 6` adds a third variant to the pilot and balances all
six execution orders. It gathers each beam's selected parent directly into the
next graph's existing KV input tensors. The returned DynamicCache is immediately
correct; the next forward copies token/position metadata but skips identical KV
tensor objects. The last unused cache and uncaptured warmup shapes retain ordinary
reordering. Duplicate-parent gathers use distinct source/destination storage;
overlapping storage is rejected. The hook restores the model on exit.

The A6000 pilot completed on four users with six balanced repetitions:

| Path | Median, ms | p95, ms |
| :--- | ---: | ---: |
| Compiled GQA | 137.178 | 142.876 |
| Staged replay | 136.332 | 143.187 |
| Direct-reorder replay | 132.010 | 138.011 |

Direct reorder improves median/p95 by 3.77%/3.40% versus compiled GQA and
3.17%/3.61% versus staged replay. Every one of the 24 paired requests is faster
than both controls. Staged replay alone slightly regresses p95 (+0.312 ms).
Direct reorder improves both metrics at both tested history lengths. Every
comparison matches all 1,200 positions, 1,200 sequence scores and 6,000 transition
scores bit for bit, with zero deltas. These are repeated four-user measurements.

Buffer and CUDA allocation/reservation remain 2.14 and 10.80/12.89 GiB. Setup
and warmup took 137.552 s; nested backend compilation was 105.097 s and graph
preparation 0.778 s. No capture/compile occurred during measurement. The audit
checks 96 direct gathers and 24 ordinary final-cache reorders during the timed
phase. [Pilot samples and audit](../benchmarks/pilots/direct-reorder) retain the
staged-replay p95 regression.

CPU checks cover the actual HF loop at both history lengths, repeated requests,
duplicate parents, exact outputs, no extra KV copy, and hook restoration. The
full-cohort confirmation adds `--full-cohort --direct-reorder --repeats 1` to the
pilot command. It preserves all 500 users, cycles the six execution orders,
and warms one representative of every length before freezing all 40 graphs.
Warmup must stay below 40 GiB allocated and 46 GiB reserved; memory failure is a
failed experiment. All three variants save 25,000 positions and sequence scores
plus 125,000 transition scores each, with source hashes and independent audits.
The full run completed and passed its independent audit.

## Full 500-user result

| Path | Median, ms | p95, ms |
| :--- | ---: | ---: |
| Compiled GQA | 136.508 | 144.643 |
| Staged replay | 134.942 | 143.908 |
| Direct-reorder replay | 131.101 | 138.838 |

Direct reorder reduces median/p95 by 3.96%/4.01% versus compiled GQA and
2.85%/3.52% versus staged replay. Every user is faster than both controls. Both
metrics improve across all ten history lengths, all six execution orders and
all five 100-user time blocks. All 500 rank lists, 25,000 rank positions, 25,000
sequence scores and 125,000 transition scores match both controls and historical
compiled GQA bit for bit. Maximum sequence/transition deltas are zero. These
are measured guarantees on this cohort, not proofs for arbitrary requests.

Setup/warmup is 145.488 s; nested backend compilation is 104.440 s and graph
preparation is 3.965 s. All 40 graphs were frozen before timing. No compilation
or capture occurred during inference. The three paths share one model, compiler
and graph pool in the paired allocation; setup is not three independent cold
starts. Previous allocations' absolute latencies are not additional gains.

The memory cost is substantial: 10.68 GiB in graph input buffers, 29.16 GiB
allocated and 34.90 GiB reserved in total. Shared input storage across mutually exclusive exact-length graphs now passes
a four-user A6000 pilot, reducing allocated memory from 10.80 to 8.99 GiB with
exact rankings and scores; full-cohort confirmation is pending.
[Input-arena results and protocol](input-arena.md) retain the small median regression.
Physical prefix sharing remains a later experiment.

[All 500 samples, independent audit and source hashes](../benchmarks/results/decode-replay)
include all three variants and order/block breakdowns. Earlier failures and
regressions remain above. This branch remains a draft experiment.
