# Shared graph input storage

The four-user A6000 pilot preserves every ranking and score bit while reducing
GPU residency. Latency is essentially unchanged. Full 500-user validation is
submitted; the pilot does not establish the full-cohort memory or latency result.

## Pilot result

FP32, TF32 disabled, beam50; four users spanning 76/121-token histories, six
trials each. All 24 rank lists, 1,200 positions, 1,200 sequence scores and 6,000
transition scores match direct reorder and historical compiled GQA bit for bit.
Maximum numerical deltas are zero. Repeated outputs are stable.

| Metric | Direct reorder | Shared input arena |
| :--- | ---: | ---: |
| Median latency | 145.348 ms | 145.465 ms |
| p95 latency | 152.721 ms | 152.254 ms |
| Unique graph-input storage | 2.136 GiB | 0.331 GiB |
| Total allocated GPU memory | 10.800 GiB | 8.994 GiB |
| Total reserved GPU memory | 12.723 GiB | 10.711 GiB |
| Peak allocated during measurement | 11.663 GiB | 9.849 GiB |

The median regresses 0.118 ms (0.081%); p95 improves 0.467 ms (0.306%). These
small differences do not establish a latency benefit. Short-history median also
regresses 0.339 ms; long-history median improves 0.609 ms. Both isolated phases
of each variant report identical allocated/reserved memory. Input storage falls
84.5%; total allocation falls 16.7%. Eight exact shapes remain resident in each
phase. The full run captures 40 shapes per phase and must be measured separately.

Cold setup/warmup takes 134.266–136.173 s per direct phase and 132.880–133.128 s
per arena phase. Nested backend compilation takes 104.205–105.656 s and
102.705–102.967 s respectively. Graph capture/warmup takes 0.842–0.848 s and
0.837–0.844 s respectively. These times overlap and must not be added. No
compilation or capture occurs during measured requests.

The initial CPU test suite found that a mock pool without capture events reported
zero logical bytes. Reporting now counts actual tensors and unique storage;
all 24 CPU/evidence tests passed before GPU submission. The GPU pilot completed
without a failed request. [Every sample, independent audit and source hashes](../benchmarks/pilots/input-arena)
are retained. These absolute latencies must not be combined with earlier runs.

## Implementation and reproduction

The direct-reorder path keeps one KV input allocation per exact-length graph.
`SharedInputArena` replaces these allocations with one flat K buffer and one
flat V buffer per layer. Each graph receives a contiguous view sized to its
original input. No attention padding, beam reduction or score changes are used.
Graph outputs and private intermediate pools remain separate.

Only one sequential model/stream is supported. A completed forward produces
independent output KV tensors; beam reorder reads those tensors and writes into
the next graph's input view. Views for different graphs intentionally overlap.
Gather sources must never overlap destinations, including for duplicate parents.
A borrowed graph output must be consumed before replaying that graph again.
This reuses input capacity across graphs; beam rows still contain dense copies
of the history.

The benchmark uses four fresh processes in one A6000 allocation:

1. Direct reorder, even cohort positions.
2. Shared inputs, even cohort positions.
3. Shared inputs, odd cohort positions.
4. Direct reorder, odd cohort positions.

Each process warms all ten lengths and captures all 40 shapes. Every one of the
same 500 users is measured once per variant. The pilot uses four users and six
repeats, split into repeats 0–2 and 3–5 in the same four-phase order. Fresh
processes make memory comparisons independent: both pools are never resident
together. Pairing is across phases, not adjacent requests. ABBA balances which
variant runs first; it does not eliminate thermal or run-to-run variability.

The report separates logical view bytes, unique input storage, total CUDA
allocation/reservation, and measurement peaks. Setup/warmup, nested backend
compilation and graph preparation are excluded from request latency. Tokenization,
checkpoint loading and output extraction are also outside request timing.

Run the pilot first, using a fresh output and cache root. In a single A6000
compute allocation with four CPU threads, run each command sequentially:

```bash
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 TORCH_LOGS=recompiles
for phase in 0 1 2 3; do
  export TORCHINDUCTOR_CACHE_DIR="$PWD/results/arena-pilot/cache/$phase/inductor"
  export TRITON_CACHE_DIR="$PWD/results/arena-pilot/cache/$phase/triton"
  test ! -e "$TORCHINDUCTOR_CACHE_DIR" || exit 1
  python -m rapidtinyrec.benchmark_input_arena \
    --release-root releases/office \
    --reference-results results/gqa/gqa_500_users \
    --output-root results/arena-pilot --phase "$phase" || exit 1
done
python -m rapidtinyrec.audit_input_arena results/arena-pilot \
  --historical results/gqa/gqa_500_users \
  --snapshot frozen-source/rapidtinyrec --run-record run-record.json \
  --output results/arena-pilot/audit.json
```

Preserve the submitted source tree and its run record as described in
[the replay protocol](decode-replay.md). After a successful pilot audit, use
fresh `arena-full` output/cache paths and add `--full-cohort` to each benchmark
command. The independent audit verifies all source hashes, cohort membership,
ABBA assignment, the full five-token FP32 sum, every score bit, repeat stability,
historical GQA outputs, actual storage bytes, and the frozen compile/capture counts.
