# Measurements

The model is the released Office MiniOneRec checkpoint: Qwen2, 1.54B parameters,
28 layers, 12 query heads and 2 KV heads. The checkpoint and data revision is
`kkknight/MiniOneRec@365a03fc6601ed36abd69cc9a0a59025a3d31cdc`.
All release hashes are in [expected.json](../rapidtinyrec/expected.json).

The cohort contains the first occurrence of each of 500 distinct Office test
users, in CSV order. Histories are 76–121 tokens. There are 3,459 catalog items
and 3,444 unique semantic IDs; recommendation metrics are at the SID level.

## Protocol

One request per user per variant; 50 beams and 50 returned sequences. FP32,
TF32 off, PyTorch 2.14.0+cu126, Transformers 4.55.4, RTX A6000, four CPU threads.
Warm every history length before timing. Synchronize CUDA around each complete
generation call. Input preparation, model loading, compilation, warmup, output
extraction and file writes are excluded. Generation setup, constraint processing,
score tensors, dense KV expansion and beam reordering are included.

Pair order alternates for the first four ablations. The GQA experiment balances
three-way order globally and within history lengths. Retain all 500 requests,
including outliers. p95 is linear interpolation at sorted index `(N-1)*0.95`.
Latency describes isolated requests across users, not concurrent serving throughput.
Replay cycles all six three-way orders across the cohort (83 or 84 users per order).

## Results and score checks

| Paired comparison | Median, ms | p95, ms | Max sequence-score difference |
| :--- | ---: | ---: | ---: |
| Released beam loop → shared prefill | 868.839 → 187.623 | 1097.674 → 189.362 | 0.000183106 |
| Shared prefill → batched constraints | 186.195 → 166.603 | 187.853 → 168.067 | 0 |
| Batched → GPU constraints | 167.792 → 160.766 | 169.247 → 163.028 | 0 |
| Eager → initial compilation | 175.161 → 171.704 | 180.492 → 185.022 | 0.000116348 |
| Initial compilation → native GQA | 174.418 → 159.687 | 188.198 → 169.783 | 0 |
| Native GQA → replay with direct KV reorder | 136.508 → 131.101 | 144.643 → 138.838 | 0 |
| Direct reorder → shared graph input storage | 130.052 → 130.240 | 137.956 → 138.305 | 0 |
| Shared inputs → physical shared-prefix KV | 129.278 → 126.697 | 137.210 → 133.267 | 0 |
| Physical prefix → folded query heads | 127.198 → 114.841 | 133.628 → 119.898 | 0 |

Each comparison preserved 500/500 full rank lists and 25,000/25,000 ranked
positions. Batched constraints, GPU lookup and GQA also preserved all 25,000
sequence scores and 125,000 transition scores bit for bit against their controls.
Shared prefill and compilation introduce floating-point rounding differences.
These are measured equivalence results on this cohort, not proofs for other inputs.

The GQA run also measured an eager control: **179.990 ms median / 185.580 ms p95**.
GQA reduced these by 11.28% / 8.51%, with identical rankings but the same rounding
differences as initial compilation. Hit@10 = 57/500 and Hit@50 = 100/500 across
all three variants.

Initial compilation slightly improved median latency but worsened p95. Tracing
showed repeated expansion from two KV heads to twelve query heads. The GQA fix
avoids that expansion for the validated unpadded attention shapes.

In the three-way run, original compilation took 187.967 s in the compiler;
GQA added 118.945 s with the same process/cache already warm. Total setup and
warmup took 320.889 s. These nested timings must not be summed. No compilation
occurred during measurement; unexpected recompilation raises an error.

Replay's full-cohort run also includes a staged-copy control: 134.942 ms median /
143.908 ms p95. Direct reorder improves median/p95 by 3.96%/4.01% versus GQA and
2.85%/3.52% versus staged replay. Every user is faster than both controls, with
improved median/p95 at all ten history lengths. All 25,000 ranks, 25,000 sequence
scores and 125,000 transition scores match both controls and historical GQA bit
for bit; maximum differences are zero.

The cost is residency: 40 graphs hold 10.68 GiB of input buffers, with 29.16 GiB
total allocated and 34.90 GiB reserved. Setup/warmup took 145.488 s; backend
compilation 104.440 s and capture/warmup 3.965 s are nested within that time.
No compilation or capture occurred during measurement. This is a sequential
inference experiment, not a concurrent serving implementation. Earlier replay
pilots include a p95 regression and a long-history median regression; see
[the full experiment record](decode-replay.md).

A separate full-cohort input-arena comparison reuses exact-shape input storage
across graphs. All 25,000 ranks/sequence scores and 125,000 transition scores
remain bitwise exact. Unique input storage falls 10.681 → 0.331 GiB; total
allocated memory falls 29.163 → 18.783 GiB and reserved memory 34.822 → 22.668 GiB.
Median/p95 regress 0.188/0.349 ms, including p95 regressions at all ten lengths.
Four fresh processes run direct/arena/arena/direct on disjoint even/odd cohort
halves, with every shape warmed per process. Memory is isolated, and latency is
paired by user across phases. This is not adjacent request alternation or a
run-to-run variability study. Cold setup/warmup is 136.887–138.923 s per phase;
all compilation/capture is excluded. [Full memory and setup record](input-arena.md).

A separate physical-prefix comparison stores one history KV row per layer and
reorders only beam suffixes. All 25,000 rank positions and sequence values,
and 125,000 transition values, match the shared-input control and historical
GQA bit for bit (maximum deltas zero). Median/p95 fall 2.00%/2.87%, improving
at all ten lengths and in both halves. Ten individual requests regress by at
most 1.059 ms; the 508.953 ms control outlier is retained.

Across the isolated phases, allocated memory falls 18.366–18.783 → 7.485 GiB,
reserved memory 22.668–22.945 → 9.313–9.779 GiB, and inference peak allocated
19.639–19.648 → 7.779–7.784 GiB. Forty graphs share 0.0145 GiB unique inputs,
compared with 0.331 GiB for the control. Native GQA still receives temporary
dense exact-length K/V; this is not a shared-prefix attention kernel. Cold
setup/warmup rises 139.682 → 349.860 s; with disk artifacts reused, setup is
69.836 → 148.810 s. All processes and GPU allocations are fresh. Compiler and
graph preparation timings overlap within setup and remain outside inference.
The four-phase ABBA run completed in 16m07s. Both earlier 15-minute timeouts
and their incomplete outputs are retained. [Full record](shared-prefix.md).

Folding the six query heads sharing each KV head into six query rows reduces
median/p95 by 9.71%/10.28% in a separate full-cohort comparison. All 500 requests
improve, by 10.100–15.703 ms; both cohort halves and all ten history lengths
improve. All 25,000 rank positions, 25,000 sequence FP32 values and 125,000
transition FP32 values match both the paired control and historical compiled
GQA bit for bit, with zero maximum deltas. All 120 saved warmups are exact too.

Allocated memory remains 7.485 GiB, reserved 9.313–9.779 GiB and inference peak
allocated 7.779–7.784 GiB for both variants. The 40 graphs retain the same unique
input storage and temporary dense K/V. Cold setup increases 346.870 → 369.764 s;
with compiler disk artifacts reused, 148.152 → 157.549 s. Four fresh ABBA
processes pair users across even/odd halves; this is not adjacent interleaving
or a run-to-run variance study. Setup and its nested compilation/graph preparation
remain outside inference, with zero measured compilation or capture. The run
completed in 20m50s. [Full exactness, memory and setup record](folded-query-heads.md).

## Evidence

Each directory in [results](../benchmarks/results) contains:

- `requests.csv`: every request's latency, output fingerprints, equality counts
  and maximum score differences. No user identifiers or histories are published.
- `summary.json`: statistics derived from those requests, environment versions,
  cohort hash and original manifest hash.
- `source.json`: original snapshot hashes and the published source hashes at the
  corresponding historical commit.

Fingerprints use SHA-256 over compact JSON token arrays and big-endian FP32 score
bytes. Full outputs are generated by the reproduction commands. The published
CSV is compact derived evidence, not a substitute for rerunning the model.

The source snapshots were measured on October 22–23, 2026 (US Eastern). Git
history was reconstructed from those snapshots. Historical commit author dates
record snapshot times; committer and GitHub PR dates record publication. Relative
imports, local output paths, FP32 defaults and launch metadata were adapted for
this repository. The original frozen sources produced the reported GPU numbers;
packaging does not constitute a new 500-user GPU run.
The later replay result was run directly from the repository sources pinned in
its evidence directory, then independently audited against the frozen snapshot.

## Scope

Shared prefill computes the history once, then **copies** its KV cache into dense
beam rows. It does not physically share storage across branches. Each wrapper
temporarily patches one model instance; do not use that instance concurrently.

The native GQA specialization assumes an unpadded single request, empty-cache
prefill and one-token cached decoding with full attention. It must not be applied
unchanged to padded batches, sliding windows or multi-token cached decoding.

BF16 shared prefill did not preserve all rankings in the earlier experiment:
only 12,990 of 25,000 positions matched. FP32 is the published correctness setting.
No claim is made against other recommendation engines, checkpoints or hardware.
