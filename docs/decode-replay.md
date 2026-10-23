# Decode replay pilot

This branch tests whether CUDA graph replay reduces the launch cost of the
compiled native-GQA decoder. The four-user A6000 pilot passed exactness checks
but produced almost no aggregate latency gain. It has not advanced to 500 users.

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

The next bounded experiment writes beam-reordered KV values directly into the
next graph's stable input buffers. This can remove the extra staging traffic;
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
