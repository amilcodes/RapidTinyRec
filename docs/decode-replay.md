# Decode replay pilot

This branch tests whether CUDA graph replay reduces the launch cost of the
compiled native-GQA decoder. There are no A6000 results yet.

The first A6000 pilot stopped during warmup, before capture or measured requests:
Hugging Face passed `inputs_embeds=None` and the argument validator rejected the
keyword. The validator now preserves that placeholder while still rejecting
actual embedding tensors. A CPU integration test runs the real constrained
50-beam generation loop through staging and compares all five scores. This fixes
the observed compatibility error; CUDA replay remains unvalidated until a rerun.

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
