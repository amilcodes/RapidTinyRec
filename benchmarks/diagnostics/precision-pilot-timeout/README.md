# TF32 pilot timeout evidence

Job 7958148 reached its 30-minute limit during phase 2 setup. Phases 0 (FP32) and 1 (TF32) completed; phase 3 never started. The original frozen auditor exited 1 because phase_2/summary.json is absent. This is an incomplete execution, not a completed performance or quality pass.

The complete fetched raw tree (235 files), submitted source/config snapshot (59 verified submitted files plus metadata), logs, generated code, API traces, original failed audit, source CI log, submission record and independent diagnostic review are preserved in hashed shards below 48 MiB. Every archived member was read back and verified. archive-manifest.json enumerates all members.

The diagnostic review calls the unchanged frozen auditor, catches its missing-file exception, and retrieves completed phase checks from that exception frame. It recomputes paired timings and four-user quality using frozen helpers. Full integrity remains failed; no missing phase is invented. This review is performed by the same research agent, not an independent reviewer identity.

In the first AB half, median/p95 generation was 114.947196/117.922076 ms FP32 versus 83.586782/84.571161 ms TF32 (27.28%/28.28% lower). There are 12 pairs across four users, three repeats each. Control is historically exact in available outputs; available within-policy checks and real model TF32 dispatch pass. Candidate scores differ; 591/600 repeated rank positions match (197/200 unique-user positions). One user's list changes. Hit/NDCG/MRR at 1/5/10/20/50 are unchanged for these four users only, with no invalid or duplicate outputs. A degenerate bootstrap interval on four unchanged user deltas does not establish population equivalence.

summary.json retains all available length, per-user, memory, dispatch, fidelity and quality details. Second-half and length-by-second-half statistics do not exist. The full predeclared performance gate cannot be evaluated, full500 eligibility is false, promotion is false and noninferiority is not established.

Completed phase setup took 758.051 and 782.936 seconds; compiler and capture durations nest inside setup and must not be summed as disjoint costs. Phase 2 had nonempty disk artifacts but still timed out in warmup. No fatal Python error or OOM is evident. Logging/compiler/capture contributions have not been isolated. Diagnose setup cost before designing a separately frozen bounded repair; no automatic replacement or full500 submission.
