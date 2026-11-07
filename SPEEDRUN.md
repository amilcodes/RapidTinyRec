# Speedrun engine

`rapidtinyrec/speedrun/` is a from-scratch inference engine for MiniOneRec (Qwen2-1.5B,
3-token semantic IDs). It does not use Hugging Face `generate`; the model forward, the
constrained search and the scoring are hand-written so that a whole request is one
CUDA-graph replay.

| file | what it does |
|---|---|
| `model.py` | Qwen2 weights from safetensors (QKV and gate/up concatenated, vocab padded to 128), RMSNorm/RoPE, prefill, shared-prefix beam step, **trie-level prefill fusion** (`prefill_tree`) |
| `beam.py` | SID catalog tables; device-resident constrained beam search (beam 50) with a fused two-token forward for the forced `\n`/EOS suffix; scoring rules `full` / `codebook` / `trie`; CUDA-graph runner with a shared memory pool |
| `exact.py` | **certified exact top-K decoding** (single-bound `ExactEngine`, three-stage `ExactEngine3`) and the exhaustive reference scorer |
| `kernels.py` | Triton shared-prefix beam attention (one program per user × row × KV head) + self-check |
| `quant.py` | weight-only INT8/INT4 (GemLite, torchao) hooks |
| `data.py` | prompts exactly as MiniOneRec's `EvalSidDataset` builds them; HR/NDCG |

Scripts: `scripts/bench.py` (latency on the frozen 500-user cohort, full-test-set quality,
throughput, exhaustive checks), `scripts/dump_scores.py` (exhaustive per-token log-probs for
every SID and user), `scripts/simulate.py` (offline search × scoring study),
`scripts/analyze.py` (tables + paired bootstrap), `scripts/oscar.sh` (SLURM runner).

## Decoders

* **Beam** (`--engine beam`): MiniOneRec semantics (beam 50, full-vocabulary log-softmax,
  forced `\n` and EOS scored). `--fuse-first` scores the prompt and all first-level trie
  nodes in one forward (exact; saves one forward).
* **Exact** (`--engine exact`, `--engine exact3`): every scored term is a log-probability ≤ 0,
  so `lp(a)+lp(b|a)` bounds every item under `(a,b)`. The engine scores every prefix, expands
  the best-bounded candidates exactly and returns a per-K certificate that the returned top-K
  is the global optimum of the model's objective.

## Example

```bash
python scripts/bench.py --release RELEASE --out results/ex3 --engine exact3 --suffix none \
  --n-prefix 64 --n-expand 50 --fuse-first --dtype bfloat16 --attn triton --compile default \
  --modes latency,quality
```

Measurements, protocol and the research log live in `research/`.
