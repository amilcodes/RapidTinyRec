# Publish fixed-prompt cache regression and audit recovery

The fixed69-token prompt cache regressed paired median latency by5.36% and p95 by5.87% on four independent users. Both lengths, both halves and all crossed cells regress. One of four rank lists changes and all sequence-score vectors differ; within-arm repeat stability passes. Job8093515 retained its original audit failure (uncached None versus zero); the distinct allocated saved-output recovery8099804 passed. Reject this candidate for speed; no full500 expansion, BF16 stacking or population-equivalence claim.

[Saved evidence, source snapshots and provenance](../research/prompt-cache-results/README.md). Publication on2026-10-31; historical work dates are retained. No adoption or automatic merge.
