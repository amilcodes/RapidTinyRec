# Physical shared-prefix cache

Experimental comparison against shared graph inputs. GPU results are pending;
no memory or latency improvement is claimed yet.

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

Within an allocated A6000 job, use separate cold compiler-cache directories for
each phase and run sequentially:

```bash
python -m rapidtinyrec.benchmark_shared_prefix \
  --release-root releases/office --reference-results results/gqa/gqa_500_users \
  --output-root results/prefix-pilot --phase 0
# Repeat for phases 1, 2 and 3, each in a fresh process/cache directory.
python -m rapidtinyrec.audit_shared_prefix results/prefix-pilot \
  --historical results/gqa/gqa_500_users \
  --snapshot frozen-source/rapidtinyrec --run-record run-record.json \
  --output results/prefix-pilot/audit.json
```

After a successful pilot, use new output/cache directories and add
`--full-cohort` to each phase. Sequential single-request inference only;
concurrent requests on the same arena are unsupported.
