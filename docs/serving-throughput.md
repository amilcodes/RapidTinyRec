# Exact-score serving throughput

Status: CPU admission prototype and proposed GPU protocol. No serving GPU result,
throughput improvement, or numerical-equivalence claim exists yet. This branch
starts from accepted folded-prefix source; it does not include rejected packed
MLP, constraint replay, beam-helper replay, or backend changes.

The user selected serving throughput while retaining exact FP32 ranks, sequence
scores, all five transition scores, both full-vocabulary normalizations and
original beam, tie and termination behavior. The first hypothesis is that two
independent requests can overlap host and GPU work without changing the shape
or arithmetic of either request. This differs from merging request rows into a
larger numerical batch, whose reduction organization would need a separate test.

## Inspiration and limits

[FlashRec's pinned architecture](https://github.com/sohu-mptc/FlashRec/blob/1089682a087ae5493f3b357ed1ec501cb684717b/docs/architecture.md)
organizes admitted requests into waves and overlaps CPU preparation with GPU
service. Its merged decode batches and native Qwen3 BF16/FP8, per-codebook scoring
are not a matched implementation of this Qwen2 FP32/full-vocabulary contract.
This experiment borrows the concurrency question; no FlashRec speedup transfers.

[PyTorch 2.14 CUDA graph guidance](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)
requires capture to be isolated from other CUDA work and warns about concurrent
replay using shared graph pools. Every lane therefore owns its model, arena,
replay inputs and outputs, and stream; capture is serial and no pool-sharing hint
is used. The [CUDA 12.6 cuBLAS reproducibility contract](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#results-reproducibility)
does not guarantee ordinary multi-stream reproducibility without additional
handle/workspace conditions. Equal shapes are motivation, not proof of equality.

The installed wheel's pinned [handle implementation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CublasHandlePool.cpp)
and [thread pool](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/detail/DeviceThreadHandles.h)
provide thread-owned handle reservations and handle/stream workspace selection.
Keep each lane's setup and requests on its persistent owner thread, verify the
installed build on compute, and audit actual results. Do not change workspace,
precision or backend settings to hide a mismatch.

## Ownership and lifecycle

`serving_lanes.LaneServer` currently implements only admission and host clocks.
It supports one or two resident runtimes, with one controller and a FIFO per
worker. Initialize each lane serially, then freeze every runtime before admission.
A lane cannot run two requests at once. Worker threads persist through setup,
measurement and cleanup. Failures preserve submitted request records, drain work
already running, skip queued work once failure is observed, and block further
admission. A runtime callback must synchronize its own stream and return fully
materialized independent CPU results before the next request reuses its storage.
The scheduler alone cannot enforce CUDA storage ownership.

The GPU runtime still needs implementation and review:

- Each lane owns a separate model, constraints, prefix arena, replay pool, prompt
  buffers, cache and HF patches. No mutable request storage is shared. Duplicate
  weights initially simplify the ownership proof; record the memory cost.
- Install the stateless process-global folded-attention patch once around the
  entire service, never patch/unpatch it independently in worker threads.
- Load, compile, warm and capture lanes serially. Complete both lanes' setup
  before freezing either compiler: the existing compiler freeze changes global
  Dynamo configuration. No compile/capture occurs during measured service.
- Use one sequential CUDA stream and thread-local inference mode per lane.
  Do not insert device-wide synchronization into concurrent timed requests.
  Keep HF's original saved-score clones and early-stop synchronization.
- Materialize final tokens, FP32 sequence/transition values and ancestry before
  releasing the request lease. Audit retained earlier results after changed and
  mixed-length requests. Preserve any mismatch, exception or lifetime failure.
- Validate disjoint live storage ranges across lanes, model parameter hashes,
  actual stream/thread identities, graph counts, replay counters and allocator
  peaks. Expected model graph counts are 8 per lane for two pilot lengths.
  Shared compiler code/cache artifacts are not permission to share tensor storage.

## Bounded pilot protocol

Primary comparison is one resident folded worker with serial admission versus
two resident folded workers with concurrent admission. Using only a two-model
serial control could overstate the improvement over the real one-worker service.
A secondary serial-admission run with both models resident isolates admission
concurrency, but cannot replace the primary comparison.

Use four frozen users: the first two at length 76 (a,b) and first two at length
121 (c,d), as selected from the historical 500-user manifest. Six simultaneous
arrival bursts each contain all four users. Orders are `(a,b,c,d)`, `(a,c,d,b)`,
`(b,a,d,c)`, repeated for the second half. This covers same/mixed-length concurrent
pairs and swaps lane ownership. Round-robin mapping is fixed within each burst.
Four fresh processes use one/two/two/one admission (ABBA), with repeats 0–2 in
half 0 and 3–5 in half 1. Disk compiler artifacts are cold in phases 0/1 and reused
by the same variant in 2/3; models, threads, streams and graphs are always fresh.
Both variants use the same four allocated CPUs and fixed two intra-op threads.

Each candidate phase additionally runs the same three bursts with two resident
models but serial admission, after primary measurements. These secondary samples
are order-confounded diagnostics, not the primary performance gate. Keep all
outputs and timing samples separate. No adaptive order, concurrency or routing.
Warm each user on every lane and repeat mixed lengths before freezing. Setup must
also establish that the one-lane worker and both candidate lanes match historical
GQA exactly; setup samples never enter throughput estimates.

Primary counts: 24 request pairs, 1,200 rank positions, 1,200 sequence values and
6,000 transitions. Compare every result against paired control and historical
compiled GQA using FP32 bits, including signed zero. Also audit all warm, repeated,
secondary-serial and retained results, exact ancestry and five original saved
score buffers. An independent auditor must recompute counts and clock metrics
from raw records, verify the schedule, lane non-overlap and concurrent host
intervals, and check graph/storage invariants. Host overlap alone does not prove
GPU overlap; an untimed two-stream profiler trace must establish actual stream
activity without being used as a performance sample.

No GPU submission until the runtime, independent auditor, provenance checks,
meaningful CPU tests and CI pass. Then one bounded regular A6000 job, four CPUs,
32 GB host RAM, at most 30 minutes. Freeze/hash source/config/resources, create
exclusive submission intent, record run/job and exact audit argv/cwd immediately,
and recover uncertain submissions instead of submitting twice.

## Measurements and advancement

The initial workload is **finite-burst core serving**, not an HTTP server or an
open-loop saturation benchmark. Prompts are pretokenized and device inputs prepared
before arrival. Every request in a burst has the same arrival timestamp. Response
latency ends after independent CPU output materialization; queue delay includes
admission and lane waiting. Report completed requests divided by wall time from
arrival to final completion, not the sum of individual service times. Never drop
failed or slow requests. Audit scans and file writes occur after the burst.

Separately report the historical generation-call boundary, now synchronized to
the request's stream. It includes model/beam work, staging and original clones,
but excludes token preparation and output/transition extraction. Concurrent
service times may rise even when burst throughput improves: neither quantity is
an isolated-request speedup. Report queue, service and response median/p95 by
length, half and burst position, all raw samples, overlap, allocated/reserved/peak
memory, logical/unique model and replay storage, and setup/model-load time. Nested
compile/warmup/capture intervals are not additive. Repeated users are not new users;
compute unique-user SID hit/NDCG/MRR separately from exact score fidelity.

A proposed predeclared pilot advancement gate is at least 5% higher primary burst
throughput overall and in each half, with response median/p95 no worse overall,
by length and by half, complete exactness and ownership stability. This is an
experimental minimum effect, not a user-provided SLO or statistical-confidence
claim. Report all strata and individual bursts, including regressions. Full500
requires a separately frozen schedule using the same source and 500 historical
users; do not submit it directly from CPU tests. Freeze that full protocol before
submission and retain all 25,000 ranks/sequence values and 125,000 transitions.
Main performance documents remain unchanged until an audited serving result
passes its gates, and serving throughput stays separate from isolated latency.


## Storage inventory preparation

`serving_storage` now describes explicitly supplied live tensors using whole
untyped-storage addresses and capacities. It unions overlapping ranges for
physical byte accounting and rejects any shared range across lane inventories,
including disjoint element views backed by the same allocation. Logical view
bytes remain separate. Empty views retain their backing allocation; expanded
views may have more logical bytes than physical capacity. Device address spaces
are checked separately.

Seven focused CPU tests cover aliasing, interval overlap, malformed records,
empty/expanded views and actual Torch views/clones in CI. This is a building
block for the CUDA runtime and auditor, not a completed ownership proof. Callers
must enumerate and retain every model/buffer, constraint, prompt, cache, arena,
graph input/output and retained score tensor while comparing both live lanes.
Graph pool identity and stream/thread ownership require separate evidence.
The CUDA runtime, complete independent auditor and GPU submission remain pending.

## Explicit owner inventory preparation

`serving_inventory.enumerate_folded_lane` enumerates registered model parameters
and buffers (including tied names), GPU constraint tables, prefix/suffix arena
backing storage and cached views, each frozen replay's input/cache/output tensors,
and explicitly named prepared prompts and retained output tensors. It returns
strong references so those storage allocations remain live during cross-lane
comparison. Enumeration does not read tensor values, copy tensors, synchronize
or execute the model, and belongs outside request timing.

The pinned Python owner schemas fail closed when fields, replay arguments or
arena identity change. Tests inject aliases into each owner category, reject
omitted/unknown owners and hidden fixed tensor inputs, and check reference
lifetime. CI additionally checks real Torch registered buffers and tied weights.
This is still preparation: it does not enumerate allocator-hidden graph-private
intermediates, prove model-specific unregistered tensor completeness, establish
stream or thread ownership, or prove exact concurrent results. Runtime integration,
graph-pool checks, stream-scoped synchronization and the independent numerical,
profile and clock auditor remain pending. No serving GPU run has been submitted.
