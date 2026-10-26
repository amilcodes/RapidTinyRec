# Full500 preparation: provenance and repeat validation

The separately frozen design covers all 500 original users in even/odd ABBA halves. Each phase has 40 graphs, 250 primary timed requests and a reverse-order untimed repeat of those same 250 users. The unchanged TF32 candidate is compared with the exact FP32 control; quality uses one primary pair per user.

`precision_full500.py` provides CPU-only scope, phase/release provenance and repeat-evidence validators. It verifies cohort content and order, phase/cache identity, expected graph/request counts, actual release digests, reverse repeat ordering, output storage separation, FP32 score sums, token/score/transition bits and ancestry. Structurally valid drift is returned with `passed=False`; untimed repeats cannot supply primary latency or inflate quality sample size.

These functions are not yet connected to the benchmark CLI or a complete full500 auditor. They do not demonstrate GPU execution, complete ownership, dispatch, source/allocation identity or runtime budget feasibility. Existing benchmark generation and timing code is unchanged. There is no full500 harness, allocation runner or submission guard, and submission readiness remains false.

The planning total is 1,580 seconds against a 1,620-second global work deadline, followed by nominal 180-second cleanup. This is not a bound. Ten-length TF32 setup, every-user repeats and allocated bootstrap costs remain unmeasured. Before submission, implement and review runtime integration, recovery-before-audit ordering, on-allocation identity, bounded audit, full source/config checks, tests/CI and an exclusive one-time guard. Full500 resampling must stay on allocated compute.

The study is exploratory and SID-level. No production loss margin is accepted, no noninferiority claim is supported, and neither this preparation nor a successful execution automatically promotes the candidate.
