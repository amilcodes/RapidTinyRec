# Decode replay pilot

This branch tests whether CUDA graph replay reduces the launch cost of the
compiled native-GQA decoder. There are no A6000 results yet.

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
```

Run on one A6000 with the pinned environment. No graph capture or compilation is
permitted after warmup. A capture failure is a failed experiment; the implementation
does not silently fall back to eager execution. CPU tests cover buffer ownership,
shape rejection and beam reordering only. A successful CUDA pilot is required
before a full-cohort run or any performance claim.

The pool supports sequential inference with one model and stream. Returned graph
tensors are borrowed until that shape is replayed again. It does not implement
concurrent serving, shared-prefix storage or variable-length graph padding.

The capture/warmup and persistent-tensor protocol follows the
[PyTorch CUDA graph guidance](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).
