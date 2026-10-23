# RapidTinyRec

Faster inference for [MiniOneRec](https://github.com/AkaliKong/MiniOneRec).
Reuse history computation across beams, keep SID constraints on the GPU, and
remove unnecessary KV-head expansion from compiled attention.

**Shared prefill: 869 ms → 188 ms median (4.63×)**, with all 25,000 ranked
positions unchanged on 500 users.

## Performance

Office checkpoint · RTX A6000 · FP32 · batch 1 · beam 50. Each row is a separate
paired run on the same 500 users; improvements should not be multiplied together.

| Change | Median, ms | p95, ms | Score equality |
| :--- | ---: | ---: | :--- |
| Prefill once, then expand KV | 868.8 → 187.6 | 1097.7 → 189.4 | Small rounding differences |
| Batch constraint transfers | 186.2 → 166.6 | 187.9 → 168.1 | Bitwise exact |
| GPU constraint lookup | 167.8 → 160.8 | 169.2 → 163.0 | Bitwise exact |
| Restore native GQA in compiled forward | 174.4 → 159.7 | 188.2 → 169.8 | Bitwise exact |

Every row preserves all 50 ranks for every user against its paired control.
Compilation and warmup are excluded. Full-vocabulary normalization, newline/EOS
probabilities, beam width and model weights are preserved.
[Measurements and limits](docs/benchmarks.md).

## Run

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
python -m rapidtinyrec.prepare_release --release-root releases/office
python -m rapidtinyrec.run prefill \
  --release-root releases/office --output-root results/prefill
```

Linux, Python 3.11+, CUDA. [Run every ablation and profiler](docs/reproduce.md).
Release files and the user cohort are pinned and hash-checked; generated outputs
include all rankings, scores and per-request timings.

[How decoding works](docs/decoding.md) · [Benchmark evidence](benchmarks/results)
· [Changes](https://github.com/amilcodes/RapidTinyRec/pulls?q=is%3Apr)

Apache-2.0. Built on MiniOneRec; see [NOTICE](NOTICE) for attribution.
