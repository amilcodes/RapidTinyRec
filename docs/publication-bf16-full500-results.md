# Publish BF16 full500 speed and quality tradeoffs

The corrected A6000 experiment8078579 passed execution and metadata audits on500 paired users. Median latency fell82.2096→56.5324ms (31.23%) and p95 fell83.5950→57.6450ms (31.04%). All500 rank lists changed. HitRate@10 was.114 in both arms, with paired95% interval for the difference[-.006,.006]; NDCG@10 changed+.000260 with interval[-.001515,+.002012]. Length91 hit-rate and length101 NDCG regressions remain in the full report. This is a measured precision tradeoff, not exact engineering or established quality equivalence. The original startup failure and pilot audit recovery remain separate.

[Saved evidence, source snapshots and provenance](../research/bf16-full500-results/README.md). Publication on2026-10-31; historical work dates are retained. No adoption or automatic merge.
