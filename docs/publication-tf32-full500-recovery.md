# Publish TF32 full500 results and separate audit recovery

The 500-user FP32/TF32 measurement completed, but job8007230 failed its allocated quality audit because the catalog path was missing. A separate saved-output audit, job8034953, passed after repairing that path; generation was not repeated. The original failure remains preserved. Descriptive paired median latency is114.4311→82.2977ms and p95 is119.5197→83.6452ms. Full quality estimates are retained in the recovered audit; no population equivalence, automatic adoption, or cross-experiment compounded speedup is claimed.

[Saved evidence, source snapshots and provenance](../research/tf32-full500-recovery/README.md). Publication on2026-10-31; historical work dates are retained. No adoption or automatic merge.
