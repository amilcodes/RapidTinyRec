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

## Recommended configurations

| goal | flags | Office, A6000 |
|---|---|---|
| MiniOneRec semantics, FP32-faithful | `--dtype float16 --fuse-first --template-cache --attn triton --compile default` | 26.1 ms, top-1 = FP32 for 500/500 |
| fastest, equal-or-better quality, certified | `--engine exact3 --norm trie --n-prefix 64 --n-expand 50 --fuse-first --template-cache --dtype float16 --attn triton --compile default` | 14.3 ms, top-50 certified 99.9% |
| best Office quality (weights tuned on validation) | same, with `--norm codebook --weights 1,0.5,0` | 14.4 ms, HR@10 0.1700 vs 0.1589 |

Other flags: `--suffix none|fused|separate`, `--norm full|codebook|trie`, `--weights wb,wc,ws`
(depth-discounted scoring), `--exit-c L` (layers for the third-code forward), `--quant gl_w8|gl_w4|fp8`,
`--beam W`, `--profile`, `--check-exhaustive K` (brute-force check of K cohort users),
`SPEEDRUN_DATASET=industrial` (MiniOneRec Industrial & Scientific).

```bash
python scripts/bench.py --release RELEASE --out results/ex3 --engine exact3 --norm trie \
  --n-prefix 64 --n-expand 50 --fuse-first --template-cache --dtype float16 --attn triton \
  --compile default --modes latency,quality
python scripts/dump_scores.py --release RELEASE --out dumps/office --shard 0 --nshards 4   # exhaustive FP32 scores
python scripts/simulate.py dumps/office --out sim.json                                   # search x scoring study
python scripts/tune_weights.py dumps/office_valid dumps/office --out tune.json           # tune on validation, test once
```

Measurements, protocol and the research log live in `research/`.
