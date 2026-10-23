# Physical shared-prefix cache

The first pilot timed out after three of four phases. The completed first-half
samples are exact and show lower memory/latency, but the required full pilot
is incomplete. A corrected bounded pilot is required before the 500-user run.

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
