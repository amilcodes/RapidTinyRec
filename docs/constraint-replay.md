# Constraint-operation replay prototype

This prototype captures the existing SID constraint processor's full-vocabulary
log-softmax, GPU lookup, mask fill/scatter and addition. It retains the upstream
normalization, all five scored tokens and the original beam selector. There is
no measured GPU correctness or latency result yet, and no performance promotion.

Existing folded-profile traces at lengths 76/101/121 show five processor calls
on contiguous `[50,152265]` FP32 scores. A score matrix occupies 30,453,000 bytes.
The prototype keeps five separate score inputs and five outputs, plus 6,000 bytes
of staged token suffixes: 304,536,000 logical input/output bytes. Captured masks,
normalization/lookup intermediates and allocator pool slack are additional and
must be measured. Existing prefix/suffix model storage remains unchanged.

Each request obtains a processor lease from `ConstraintReplayPool.processor()`.
It stages the scores and last three integer token IDs, then replays the graph
for that step. Five independent graph-private pools keep each saved score output
alive through subsequent steps. The caller must consume transition-score and
other saved-score outputs before calling `lease.close()`; overlapping request
leases and use after close are rejected. The pool supports one sequential CUDA
stream, one score shape per step, and precisely the five validated steps.

Preparation warms and captures the same ATen operations without `torch.compile`.
`freeze()` requires all five graphs and no live lease. Capturing after freeze,
changing score layout/dtype/device, and using another stream are errors. Both
score copies and integer suffix copies remain inside generation-call timing.
Buffer reuse is an explicit borrowing contract, not general concurrent serving.

CPU checks compare the unchanged operations against the existing processor for
valid and unknown prefixes, test staging ownership with mixed input lengths,
and verify lease lifecycle guards:

```bash
python -m unittest discover -s tests -p test_constraint_replay.py -v
```

Before a GPU submission, integrate a fresh-process folded-control/candidate ABBA
benchmark and independent auditor. The four-user pilot must check every warmup
and repeated output against paired controls and historical compiled GQA, five
independent score outputs, graph/direct equality, unknown-prefix fallbacks and
mixed-length safety. Freeze eight model graphs plus five constraint graphs per
candidate phase; no timed capture or compilation. Include staging and all work
inside `generate`; input preparation, output extraction and file writes are
outside the existing timing boundary. Record isolated allocated/reserved and
peak memory, logical/unique graph storage and prefix/suffix capacities. Setup
contains nested compilation/warmup/capture intervals and must not be summed.

Only a complete exact and stable pilot with useful median/p95 improvement by
length and half can authorize the same-source frozen-500 comparison. That run
uses forty model graphs plus five constraint graphs per candidate phase and
audits all 25,000 ranks, 25,000 sequence values and 125,000 transitions. Main
performance documentation stays unchanged until full validation.

This isolated branch starts from the validated folded/profile branch (#12).
The rejected wide and batched packed-MLP evidence remains in #13. Graph-resident
constraint work is inspired by FlashRec's orchestration, but this prototype
does not adopt its restricted output-head normalization or precision modes.
