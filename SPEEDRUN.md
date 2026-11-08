# Speedrun engine

`rapidtinyrec/speedrun/` is a from-scratch inference engine for MiniOneRec (Qwen2-1.5B,
3-token semantic IDs). It does not use Hugging Face `generate`; the model forward, the
constrained search and the scoring are hand-written so that a whole request is one
CUDA-graph replay.

| file | what it does |
|---|---|
| `model.py` | Qwen2 weights from safetensors (QKV and gate/up concatenated, vocab padded to 128), RMSNorm/RoPE, prefill, shared-prefix beam step, **trie-level prefill fusion** (`prefill_tree`) |
| `beam.py` | SID catalog tables; device-resident constrained beam search (beam 50) with a fused two-token forward for the forced `\n`/EOS suffix; scoring rules `full` / `codebook` / `trie`; CUDA-graph runner with a shared memory pool |
| `exact.py` | **certified exact top-K decoding** (single-bound `ExactEngine`, three-stage `ExactEngine3`, `ExactTrieEngine` that skips single-child prefixes, `OnePassTrieEngine` that scores every item in one late-entry forward) and the exhaustive reference scorer |
| `bridge.py` | **late-entry bridge**: predicts a prefix row's residual stream at layer e from its parent node's state (ridge + distilled MLP) |
| `kernels.py` | Triton shared-prefix beam attention (one program per user × row × KV head), causal prompt attention for the fused first forward, self-checks |
| `skinny.py` | split-K GEMM for skinny activations (benchmarked; not used by default) |
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
* **Exact, trie** (`--engine exacttrie`, `--norm trie|cbtrie`): under a trie-normalized third code
  single-child prefixes are exact after the first forward, so the second forward expands only the
  M best multi-child prefixes (M=16 certifies top-10/20; M=256 certifies top-500).
* **Late entry** (`--late-entry E:bridge.pt`): the second forward runs only layers E..27; its input
  is the bridge applied to the parent node's layer-E state. Bridges are trained on validation
  prompts with `scripts/train_bridge.py` (≈1–2 min on one H100, no labels).
* **One pass** (`--engine onepass`, needs `--late-entry` and `--tree-fast`): every multi-child prefix
  joins the first forward at layer E; every item is scored in one forward (exhaustive; slower than
  two forwards at batch 1).

## Recommended configurations

| goal | flags | Office, A6000 / H100 |
|---|---|---|
| MiniOneRec semantics, FP32-faithful | `--dtype float16 --fuse-first --template-cache --attn triton --compile default` | 26.1 / 8.5 ms, top-1 = FP32 for 500/500 |
| certified, no learned parts | `--engine exacttrie --n-prefix 16 --n-expand 50 --fuse-first --template-cache --dtype float16 --attn triton --tree-fast --compile default` | 13.3 / 4.6 ms, top-10 certified 100% |
| **fastest certified** | same + `--late-entry 22:bridge_e22_w.pt` (+ `--compile-layer` on the H100) | **9.1 / 3.2 ms**, HR@10 = full depth |
| best Office quality | `--engine exact3 --norm codebook --weights 1,0.5,0 --n-prefix 64` + the late-entry flags | 9.7 / 3.8 ms, HR@10 0.170 vs 0.159 |
| large candidate lists | `--engine exacttrie --n-prefix 256 --n-expand 500 --ks 100,200,500` + late entry | — / 3.8 ms, top-500 certified 100% |
| **fastest, Office (third session)** | `--engine exacttrie --n-prefix 16 --n-expand 50 --norm cbtrie --weights 1,0.75,0 --tree-fast --compile-layer --fuse-store --pdl --combo --late-entry-a 18:bridge_b_e18_w.pt --late-entry 26:bridge_e26_cas.pt` | 6.4 / **2.75 ms**, lossless vs full depth |
| **best quality, still fast** | same + `--bonus-file sasrec_test.pt --bonus-beta 0.5` (per-user β·log q_SASRec, in the graph) | 6.4 / **2.72 ms**, HR@10 0.1825 |
| throughput | same + `--quant fp8` (dynamic per-token FP8, quality-neutral) | 3,155 users/s at B=128 (H100) |
| Industrial | bridges at 18/18 (`casInd`), entry 26 is not lossless there | — / 3.11 ms (3.45 with fusion) |

Deepest lossless third-code entry is model-dependent (released Office: 26; fine-tuned Office: 22; Industrial:
18): choose it on validation. New flags: `--bonus-file/--bonus-beta` (per-user item term), `--pdl`, `--combo`,
`--engine inflight` (negative), `--fused fp16|int8` (negative). Scripts: `sasrec.py`, `fuse_search.py`
(fusion; `--large-k`, `--wc-grid`), `value_search.py`, `filter_search.py`, `lbound.py` (two-sided bounds),
`train_sft.py` (model branch), `paired.py` (paired bootstrap between two quality runs).

Other flags: `--suffix none|fused|separate`, `--norm full|codebook|trie|cbtrie`, `--weights wb,wc,ws`
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
python scripts/train_bridge.py --train-release VALID_RELEASE --release RELEASE --out bridges \
  --entry 22 --row-weight 0.5 --epochs 6 --train-users 4600 --tag _w                 # late-entry bridge
```

Measurements, protocol and the research log live in `research/`.
