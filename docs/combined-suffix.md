# Combined suffix scoring (experimental)

Combining the last two transformer forwards reduced paired median latency from
134.523 to 129.310 ms and p95 from 142.701 to 137.443 ms on 500 users. All 25,000
rank positions matched. Scores have rounding differences, so this remains an
experimental option rather than a bit-exact replacement.

Process `[third SID token, newline]` in one forward. The third-token query cannot
attend to newline; the newline query sees the whole cache. Two single-query native
GQA attention calls preserve that visibility, while projection and feed-forward
work operate on both positions together. Save the second position's EOS logits,
reorder them with the selected beam parents, and return them on the next generation
step. Hugging Face still performs five scoring/selection steps; model forwards
drop from five to four.

```bash
python -m rapidtinyrec.test_suffix_smoke
python -m rapidtinyrec.run suffix --release-root releases/office \
  --reference-results results/gqa/gqa_500_users --output-root results/suffix
python -m rapidtinyrec.audit results/suffix/suffix_500_users --output results/suffix/audit.json
```

The paired A6000 run used FP32, TF32 off, the pinned checkpoint/cohort and beam 50.
All 500 candidate sets, top-1 results and full rank lists matched the GQA control.
Each saved sequence score equals the sequential FP32 sum of all five transition
scores. Hit@10 = 57/500 and Hit@50 = 100/500 for both variants.

| Score | Bit-identical values | Maximum absolute difference |
| :--- | ---: | ---: |
| Sequence | 6,506 / 25,000 | 0.000053405762 |
| SID1–3 transitions | 75,000 / 75,000 | 0 |
| Newline | 2,474 / 25,000 | 0.000030279160 |
| EOS | 6,694 / 25,000 | 0.000030755997 |

No user's complete score vector is bit-identical. Batching two token positions
changes projection/MLP shapes and floating-point rounding; the observed drift
is confined to the two affected suffix probabilities. The attention oracle
also confirmed causal isolation and newline self-attention. This preserves the
scoring formula, but identical rankings on this cohort do not guarantee identical
rankings on every input.

Every user became faster (median paired saving 5.161 ms). Median and p95 improved
within all ten history lengths, both pair orders and all five 100-user blocks.
The rerun GQA control matches its historical output bits exactly, but its latency
is lower than in the earlier allocation. The measured suffix gain is 3.87% median
and 3.68% p95; cross-run latency differences are not added to it.

Setup and warmup took 258.545 s. Compiler time was 135.792 s for GQA and an
incremental 114.208 s for combined suffix after GQA warmed the process/cache.
These are overlapping measurements, not independently cold runs. There were no
compilations during inference. Three expected compiler tracing restarts completed
successfully; no requests or outliers were removed.

[Per-request fingerprints, score mismatches, grouped latency and audit](../benchmarks/results/combined-suffix)
are published alongside source hashes. Original frozen sources produced these
measurements; the portable package was checked separately with CPU functional
tests covering attention, 50 rankings, scores, forward counts and beam reordering.
