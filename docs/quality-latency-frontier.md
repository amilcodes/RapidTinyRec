# Quality–latency research track

## Complete paired pilot passed, October26

The complete four-phase pilot finished in18m17s and passed its frozen fidelity, stability, ownership, dispatch, validity and performance gates. Paired median/p95 generation latency fell **115.576/118.961 → 82.957/83.655ms** (28.22%/29.68% lower), with gains across both lengths and both halves. Candidate scores differ and one user list changes; all four users have unchanged Hit/NDCG/MRR. Four users do not establish quality parity or noninferiority. The result supports designing a separately frozen full500 review, with no automatic submission or promotion.

[Complete paired evidence, memory, uncertainty and538-member archive](../benchmarks/precision-paired-staged-pilot/README.md). The original timeout and TF32-only diagnostic remain preserved below; their earlier next actions are historical.

## Historical setup diagnostic, October26

The separately frozen TF32-only diagnostic completed in9m22s and passed its audit. Cold/reuse setup took357.409/152.681seconds; all1200ranks,1200sequencevalues,6000transitions and ancestry match preserved TF32 outputs exactly. Ownership/stability, actual graph-linked dispatch and hashed log recovery pass. [Complete setup evidence and limitations](../benchmarks/diagnostics/precision-setup-feasibility/README.md) includes310verified archived members.

This establishes setup feasibility only. There is no paired FP32 control, new quality estimate or causal logging claim; full500/promotion/noninferiority remain false. The later complete paired pilot is reported above. Its eligibility is separate from this diagnostic. Earlier preparation descriptions below are historical.

The user explicitly authorized investigating large inference gains even when
numerical outputs differ, provided recommendation quality is unchanged or only
slightly affected. This is a separate research track from exact-score serving
(PR17). The exact reference, original audits and rejected results remain intact.
A failed bitwise comparison is a fidelity result, not by itself evidence of worse
recommendations. Conversely, small score deltas or high beam overlap do not prove
that held-out recommendation quality is preserved.

Branch: `experiment/quality-latency-frontier`, based on accepted
`experiment/12-folded-profile` at `4711e463321deadb835ac8cc25c1dfbf41d86108`.
The paired quality evaluator and one TF32 matmul candidate are implemented.
Job **7958148** timed out during phase 2 setup after completing the first FP32/TF32 half. The original frozen audit failed on missing `phase_2/summary.json`; full500 eligibility remains false. Partial median/p95 were **114.947196/117.922076 ms → 83.586782/84.571161 ms** (27.28%/28.28% lower). Available control/stability/ownership/dispatch checks pass. Candidate scores differ and one of four users has a changed list; all four users’ Hit/NDCG/MRR metrics are unchanged. Four users and an incomplete ABBA schedule cannot establish quality parity or a passed performance gate. Completed setup took 758.051/782.936 seconds; diagnose setup cost before any separately frozen bounded repair. No automatic resubmission.

Complete failure evidence: [timeout archive](../benchmarks/diagnostics/precision-pilot-timeout/README.md). The exact serving track remains separate.

## First opportunities

1. **Reduced internal matmul precision with FP32 storage/output.** Design a single
   explicit TF32-enabled candidate against the original FP32/TF32-off control,
   on the same checkpoint and A6000. Keep beam 50, five score steps, the full
   vocabulary, both normalizations, constraints and termination unchanged. This
   isolates numerical precision before changing the scoring objective. Confirm
   installed PyTorch policy and actual kernel dispatch; do not identify a getter
   with acceleration. [PyTorch 2.14 documents internal precision independently
   from output dtype](https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html).
   Its `high` policy may select different implementations; pin the exact setting
   and record what runs, without an algorithm/workspace sweep.
2. **BF16 inference** is a follow-on hypothesis if source/profile review suggests
   substantially greater benefit. Specify accumulation and logits/normalization
   casts explicitly, measure their cost, and test this change separately. A
   checkpoint converted at inference is not a retrained checkpoint.
3. **Merged request batches and selective-head scoring** may offer structural
   gains, but must be distinct later candidates. Batch shapes can alter rounding;
   restricted normalization can change cross-prefix beam ordering even if each
   local ordering appears unchanged. Preserve invalid-output, termination and
   catalog checks and disclose every semantic difference.

[FlashRec's pinned documentation](https://github.com/sohu-mptc/FlashRec/blob/1089682a087ae5493f3b357ed1ec501cb684717b/README.md#numerical-match-vs-huggingface)
separates BF16/FP8 numerical overlap from retrieval quality. Its Qwen3 workloads
and native codebook scoring are not a head-to-head result for our Office/Qwen2
checkpoint. Use its design as a hypothesis, then measure on common hardware,
checkpoint, users and catalog. Native FP8 support must be verified for this
hardware/software before proposing an experiment; no headline speedup transfers.

Do not automatically rerun old candidates. The cuBLASLt down path and batched MLP
were slower, so relaxed fidelity does not rescue their performance evidence.
The old wide-MLP pilot's four-user unchanged ranks and modest timing benefit are
insufficient quality evidence and lower priority than a large hardware opportunity.
Reuse archived outputs for analysis before considering any new run.

## Quality evidence and decisions

Use the same frozen users, history construction, targets, catalog, token map,
SID-to-item mapping, tie handling and output count for paired comparisons. Keep
numerical fidelity (bit matches, score deltas, rank/beam overlap) separate from
held-out target quality. Report hit rate, NDCG and MRR at 1/5/10/20/50, with primary
NDCG@10 and Hit@10 declared before viewing candidate results. Report per-user
paired changes, improvements/regressions, length strata, invalid and duplicate
recommendations, coverage, and memory. Existing metrics are SID-level; do not
rename them item quality if SID collisions or item mapping are unresolved.

Use user-level paired confidence intervals with a fixed resampling protocol and
seed; repeats measure runtime stability, not additional quality samples. A tiny
pilot is for execution and latency feasibility, never population quality parity.
A promising pilot may advance to the frozen 500-user quality evaluation despite
nonidentical scores. Evaluate both control and candidate on those users; preserve
all ranks, sequence scores and five transitions for the initial precision-only
candidate, including every failed exactness comparison. Bit equality is recorded
but is not this track's promotion gate. Complete output validity, ownership and
repeat stability remain mandatory.

The initial quality target is no aggregate observed loss in primary metrics and
no concentrated serious regression in a subgroup. "No statistically significant
loss" is not a noninferiority result. The user has not supplied a numerical loss
budget. Propose an explicit noninferiority margin, justified in interpretable
metric units and frozen before confirmatory analysis; do not silently invent an
accepted production tolerance. If confidence intervals cannot exclude a material
loss, call the result inconclusive and propose a larger/disjoint held-out check.
Repeated use of the same 500 users for candidate selection is exploratory; reserve
a disjoint confirmatory set if comparing several candidates. No retraining.

Rank candidates by the measured quality–latency–throughput–memory frontier. Seek
material gains, not tiny noisy wins in exchange for changed behavior. A promising
research result does not automatically change the accepted exact baseline or
merge a branch. Present concrete gain/loss and uncertainty before adoption.
Research pilots do not need renewed permission merely because bits differ.

## Execution discipline

One candidate at a time across both tracks, one regular A6000, four CPUs/32 GB,
maximum 30 minutes per job. Source/tests/CI and an independent quality/fidelity
and performance auditor before a bounded paired pilot. Fresh processes and
counterbalanced order; identical timing boundaries and workload. Report setup
and model load separately, no overlapping-time sums or cross-run compounded
speedups. Single-request latency and serving throughput are different outcomes;
serving response includes queueing under its defined load.

Freeze/hash all submitted source/config/resources and preserve raw results,
logs, audits, source snapshots and reproduction scripts. Guard submissions with
intent/run records and recover uncertain outcomes instead of duplicating jobs.
Preserve execution failures and diagnose them; no blind replacement or unbounded
precision/algorithm/beam sweep. No paid infrastructure, subagents or merges.

## Paired evaluator implementation

`python -m rapidtinyrec.quality_frontier INPUT.json --output REPORT.json` takes
explicit unique users (`cohort_index`, `prompt_tokens`, three `target_token_ids`),
observations (`cohort_index`, `variant` control/candidate, `repeat`, original 50
five-token `token_ids` rows), unique catalog SIDs and `eos_token_id`. Inputs must
come from a separate provenance/fidelity audit; the evaluator does not certify
checkpoint, token-map or target hashes. Reports hash their input and refuse to
overwrite existing evidence.

Optional `target_item_count` metadata must cover every user if supplied; it adds
ambiguous/unambiguous target strata without removing users. The caller must verify
the mapping provenance. The evaluator validates complete user/repeat pairing and reports token-rank
repeat instability instead of silently selecting a favorable repeat. Quality
uses the lowest matched repeat once per user. Invalid rows retain their rank
positions and cannot hit the target; duplicate valid SIDs consume positions and
are reported, not deduplicated into an inflated score. The existing release's
suffix is newline token198 followed by supplied EOS; another task must explicitly
revise that schema. SID set overlap is a diagnostic even for invalid outputs.
No numerical-score stability is inferred from stable tokens.

Freeze bootstrap protocol at 10,000 user-level paired draws, seed20261026,
two-sided percentile95% with linearly interpolated endpoints. Length strata use
seed20261026+prompt_length and have unadjusted descriptive intervals. Primary
metrics remain Hit@10/NDCG@10; other cutoffs and strata are exploratory. The report
includes per-user ranks/changes and improved/regressed/unchanged counts, so a zero
aggregate change cannot hide swaps between winners and losers. Even an all-zero
interval does not establish population noninferiority. `promotion` and
`noninferiority_established` remain false; this module never auto-accepts a model.
Large-cohort resampling belongs on allocated compute; local/CI tests use tiny
fixtures and 64 draws. No GPU workload or large dataset is run by these tests.

## Catalog identity finding

A bounded metadata audit of the released catalog found **3,459 item IDs mapped to
3,444 SIDs**: 15 SID collisions, each covering two items. **Seven of the frozen
500 users have a target SID with two possible item IDs.** Index/item resources
were newly hash-pinned; all SID/item-ID associations were cross-checked against
the already release-pinned catalog file, and target identities/source row indices
against the already pinned test CSV and cohort manifest. The [mapping review](../benchmarks/preparation/quality-frontier/catalog-mapping-review.json)
preserves hashes, every collision and the affected cohort indices.

SID-level quality therefore cannot simply be relabeled as exact item retrieval
quality. Item IDs are recoverable from the test data, but model outputs alone do
not resolve a colliding SID. Keep all 500 users in primary SID metrics; report
ambiguous-target and unambiguous-target strata separately when integrating the
quality auditor. Do not drop those seven users or select the favorable member of
a collision. An eventual item-level comparison needs a fixed, common expansion
or disambiguation policy with output-count/rank effects accounted for. This
metadata finding is not evidence of any candidate regression.

## TF32 pilot implementation and frozen decision rule

`benchmark_precision.py` implements fresh-process folded-FP32 / folded-TF32 /
folded-TF32 / folded-FP32 ABBA. Both sides retain FP32 model parameters, logits,
saved scores, transitions, original full-vocabulary normalizations, beam50 and
termination. The only numerical policy difference is the legacy
`torch.backends.cuda.matmul.allow_tf32` setter before compilation and capture;
cuDNN TF32 stays disabled. This deliberately uses the same API as the accepted
control and does not mix hierarchical precision setters. In the pinned wheel's
[Context.cpp](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/Context.cpp#L368),
this setter updates both legacy HIGH/HIGHEST and CUDA MATMUL TF32/IEEE policy.
The effect can include prefill, attention matmuls and the output head, not just
MLP projections. It does not promise that every operation uses tensor cores.

The source review pins eight installed-wheel files, including the Inductor native
`mm.out` selection, precision setters and cuBLAS handle math-mode selection.
Runtime rejects build drift, precision/backend/workspace environment overrides,
TunableOp and Inductor max-autotune. It leaves backend/workspace defaults intact.
The benchmark saves generated wrapper/kernel sources, native API logs during
untimed compiled warmup/capture, and post-measurement frozen model replay traces
for both lengths. Dispatch verification requires original decode gate/up/down
GEMM shapes with the intended API math mode plus corroborating kernel symbols
and CUDA graph launches. A getter, unrelated tiny GEMM or kernel name alone
cannot pass. Missing or ambiguous dispatch evidence blocks advancement. This
is not an instruction-level disassembly proof or a measured acceleration claim.

The pilot uses four frozen users (first two each at lengths76/121), six paired
trials, eight model graphs per phase, cold disk artifacts in phases0/1 and
same-variant artifact reuse in2/3 with fresh allocations/processes. Timing has
identical boundaries on both sides. API logging, profiling, output extraction,
retained-output scans and files are excluded; staging and original HF clones are
inside generation. Model loading is separate. Setup intervals nest.

Before seeing results, require **at least10% lower generation median and p95**
overall, in each length and in each ABBA half, with no length-by-half median/p95
regression, to justify a full exploratory quality evaluation. This is a research
minimum-effect rule, not an accepted user SLO or statistical confidence claim.
Also require valid/unique outputs, exact historical FP32 control, stable
within-policy repeated/graph/direct/retained outputs and ancestry, frozen graphs,
independent original saved HF score storage, and verified runtime/dispatch.
Candidate/control FP32 differences remain reported rather than rejected.

`audit_precision.py` keeps numerical fidelity separate from the paired SID
quality evaluator, including ambiguous targets. Invalid or duplicate outputs
remain in their original rank positions and fail validity; they are not removed
from quality estimates. A four-user pilot never establishes population parity.
The auditor never promotes a candidate or declares noninferiority. Full500
bootstrap must run on allocated compute; the audit refuses that resampling on a
local machine. The first pilot timed out; its incomplete evidence is preserved in the timeout archive.


## Timeout repair preparation: local diagnostic logging

The proposed repair moves only the native API log sink to verified node-local
storage. It preserves the logging level, capture coverage, numerical operations,
cohort, beam/scoring semantics and timed generation boundary. Logging overhead is
still a hypothesis: original cold setups consumed 758.05/782.94 seconds and wrote
51.31/51.67 MB of API logs. Compiler and capture durations overlap inside setup;
they must not be summed. No repaired GPU run or acceleration claim exists yet.

An explicit `--api-log-local-root` uses checked logger disable/flush followed by
hashed, atomic log publication outside measurement. Setup exceptions preserve
partial logs; copied hashes and exclusion from measurement are audited. Original
local bytes remain until the supervisor observes child exit. The supervisor uses
a private child process group, a maximum 1,620-second child deadline, TERM/KILL
escalation, and checked recovery to leave time within a 30-minute allocation.
Conflicting destination evidence is never overwritten. A forced kill can lose
native userspace buffers; killing the supervisor or losing the node cannot
guarantee recovery. Missing evidence continues to block acceptance.

Twenty CPU tests cover staging, failure/partial recovery, copy conflicts, path
confinement, logger-disable/flush failures, actual child timeout and forwarded
signals. These tests do not establish CUDA behavior or a 30-minute ABBA budget.
The original failure snapshot and archive are unchanged.

Next: review a separately bounded setup feasibility diagnostic with its own
profile, audit and one-time submission guard. A single-policy setup measurement
would establish feasibility only, without replacing a complete paired ABBA
pilot or qualifying for full500. Submission remains blocked until the concrete protocol, source review, tests
and budget are ready under the existing research authorization.


## Distinct setup-feasibility protocol

The [frozen protocol](../benchmarks/preparation/quality-frontier/setup-feasibility-protocol.json)
selects TF32 phase 1 cold setup followed by phase 2 disk-artifact reuse, each in a
fresh process with the original four users, eight graphs and twelve request rows.
The [allocation script](../benchmarks/preparation/quality-frontier/precision-setup.sh)
uses one A6000, four CPUs and 32 GB for at most 30 minutes, with a 1,620-second
supervised child deadline. The remaining nominal 180 seconds covers termination
and recovery; it is not a guaranteed cleanup duration. No new GPU job is submitted.

`audit_precision_setup` requires both phases, normal supervised exit, complete
hash-matched log recovery, the original source/runtime/resource/graph/ownership
checks, actual TF32 dispatch and exact within-policy output/ancestry stability.
Every request must also match preserved original TF32 outputs; this checks that
the logging repair itself leaves numerical behavior unchanged. FP32 differences
are retained descriptively. Successful setup validation always leaves full500
eligibility, promotion and noninferiority false. The complete ABBA auditor rejects
this diagnostic scope. Eight CPU tests exercise these decision boundaries and
mutations of real log publication/recovery records.

Cold/reuse setup, nested compiler/capture intervals, memory and generation
latencies will be reported separately. There is no concurrent FP32 control, so
this diagnostic cannot establish a speedup, paired recommendation quality or a
causal estimate of logging overhead. The previous incomplete ABBA evidence stays
separate. Before submission, a distinct harness, immutable source/config snapshot,
one-time submission guard and passed publication CI must be recorded. Review the
resulting budget before designing any later complete paired pilot.


### Historical complete paired pilot preparation

A distinct four-user FP32/TF32/TF32/FP32 pilot is being prepared after the successful setup diagnostic. The frozen [design](../benchmarks/preparation/quality-frontier/precision-paired-staged-protocol.json) plans 1500 seconds of child work under the existing 1620-second deadline, leaving nominal cleanup time within the 30-minute allocation. Repaired-logging FP32 setup remains unmeasured; this is a planning estimate, not a completion guarantee.

The new phase manifests serialize the actual eleven release hashes already verified before model loading, including the checkpoint, and bind each phase to the frozen protocol. The previous setup benchmark already checked these hashes; its limitation was the absence of a serialized observed digest record. The distinct paired auditor additionally requires all four phase records and complete wrapper recovery before applying the original paired fidelity, dispatch, quality and performance gates. Generation, extraction and timing-loop ASTs remain identical to the original pilot. This source preparation has not submitted a new job: harness configuration, a one-time submission guard, source CI and final preflight review remain required. No full500 or promotion is implied.

## BF16 against the audited TF32 baseline

The next isolated experiment is `bf16-vs-tf32-pilot`, with fresh-process
TF32/BF16/BF16/TF32 ordering and same-variant disk compiler reuse in the second
half. The four frozen users and six paired trials retain beam 50, all five scored
tokens, full-vocabulary normalization, HF output ownership, and the existing
generation timing boundary. This is a pilot, not a population quality claim.

The candidate loads BF16 weights and uses BF16 activations, KV storage and model
logits. HF retains its existing copy to FP32 before normalization and beam-score
accumulation. CUDA TF32 remains enabled for residual FP32 matmuls in both policies;
cuDNN TF32 is off. BF16 reduced-precision accumulation is disabled, with the
existing split-K permission preserved. There is no autocast or backend sweep.

The exact control reference is the saved TF32 output from job 8007230, whose
separate allocated recovery audit passed in job 8034953. The original generation
job's post-generation audit failure is preserved. The pilot checks the recovered
audit hash and saved raw TF32 row hashes rather than treating a failed original
job as a completed audit. Catalog resource hashes are checked before model load.

Dispatch evidence requires native BF16 operands and FP32 compute for actual MLP
shapes, corroborated by BF16 kernels linked to the captured model graphs. Getter
values or unrelated BF16 probe kernels cannot pass. Both policies must retain
direct/replay/repeat/profile stability and independent FP32 score ownership.

The run has a separate profile, frozen protocol, allocation identity, 1,620-second
global work deadline and 1,500-second generation bound within one regular
A6000/4-CPU/32-GB/30-minute allocation. Logs and failed evidence are preserved.
A material incremental effect is screened at 10% median and p95 improvement
overall, by length and by half, with no length-by-half regression. A valid
material pilot can advance to a separately frozen 500-user experiment under the
user's standing authorization. No promotion or noninferiority claim is automatic.
