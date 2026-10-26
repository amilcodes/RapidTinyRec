# Full500 preparation: provenance and repeat validation

The separately frozen design covers all 500 original users in even/odd ABBA halves. Each phase has 40 graphs, 250 primary timed requests and a reverse-order untimed repeat of those same 250 users. The unchanged TF32 candidate is compared with the exact FP32 control; quality uses one primary pair per user.

`precision_full500.py` provides CPU-only scope, phase/release provenance and repeat-evidence validators. It verifies cohort content and order, phase/cache identity, expected graph/request counts, actual release digests, reverse repeat ordering, output storage separation, FP32 score sums, token/score/transition bits and ancestry. Structurally valid drift is returned with `passed=False`; untimed repeats cannot supply primary latency or inflate quality sample size.

The benchmark CLI now requires explicit `--full-cohort --full500-quality` with staged logging and exclusive scope. It records frozen full500 provenance, freezes primary timing/memory summaries, then saves each reverse-order untimed repeat before checking stability. The audit core reads those repeat files, validates frozen graph/reorder activity and includes their bit/ancestry comparisons in the stability gate. Original generation, extraction, memory and timed-loop AST boundaries remain unchanged.

The complete allocated-audit entry point, source/allocation identity binding and deadline/recovery runner remain pending. CPU fixtures do not demonstrate GPU execution, complete ownership, dispatch or runtime budget feasibility. There is no full500 harness or submission guard, and submission readiness remains false.

The planning total is 1,580 seconds against a 1,620-second global work deadline, followed by nominal 180-second cleanup. This is not a bound. Ten-length TF32 setup, every-user repeats and allocated bootstrap costs remain unmeasured. Before submission, implement and review runtime integration, recovery-before-audit ordering, on-allocation identity, bounded audit, full source/config checks, tests/CI and an exclusive one-time guard. Full500 resampling must stay on allocated compute.

The study is exploratory and SID-level. No production loss margin is accepted, no noninferiority claim is supported, and neither this preparation nor a successful execution automatically promotes the candidate.
