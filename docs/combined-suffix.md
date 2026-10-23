# Combined suffix scoring (experimental)

This branch combines the last two transformer forwards while retaining both
newline and EOS probabilities. There is no 500-user A6000 performance or exact
score-equivalence result yet.

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

The tiny random CPU model checks the attention rule, 50 returned rankings,
score differences, forward counts and beam reordering for three history lengths.
It cannot establish production latency or full-checkpoint FP32 equivalence.
Batching two tokens can change floating-point rounding even when the scoring
formula is preserved. The A6000 harness retains every score for that comparison
and records compile/warmup time separately.
