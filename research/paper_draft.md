# Beam Search Is the Wrong Abstraction for LLM Generative Recommenders
### Certified exact decoding, scoring semantics, and a 30–70× inference speedrun for semantic-ID recommendation

*Draft — numbers marked [TBD] are pending runs; every number cites a run directory under `runs/`.*

## Abstract (draft)

LLM-based generative recommenders produce items by beam-searching short semantic-ID (SID)
sequences under a catalog trie. We study the inference of an open, widely used instance
(MiniOneRec, Qwen2-1.5B, 3-token SIDs, beam 50) end to end. (1) A roofline analysis shows that
batch-1 latency is set by the *number of sequential forwards* times *weight bytes*, not by beam
width, so beam search spends most of its time re-reading weights to explore a tree whose shape
is known in advance. (2) Exploiting that the trie is tiny (88 first codes, 2,488 two-code
prefixes for 3,459 items) and that every scored term is a log-probability ≤ 0, we give a
*certified exact* decoder: two tree-structured forwards score every first code and every
prefix, a third teacher-forces the best-bounded candidates, and an admissible bound proves
when the returned top-K is the global optimum of the model's own objective. (3) Running exact
search reveals that MiniOneRec's full-vocabulary objective — including the forced newline/EOS
tokens that production engines silently drop or renormalize away — is mis-calibrated: closer
to the true argmax is *worse* for HR@10, while dropping the forced suffix or renormalizing over
the trie improves quality *and* makes certification succeed for most users. (4) Combined with a
hand-written engine (one CUDA graph per request, fused beam attention, weight-only INT8), we go
from 869 ms (HF `generate`, FP32) to [TBD] ms per request on one RTX A6000 at equal or better
HR/NDCG.

## 1. Introduction
- GR is being deployed (OneRec, GR4AD, GRACE, PinRec, Spotify, NVIDIA SID-GR); serving cost
  is the bottleneck; engines report throughput with approximate parity and undocumented scoring
  choices.
- Our questions: where does the time go at batch 1? Is beam search necessary? Which score is
  being optimized, and does exactness help?
- Contributions (see `PLAN.md`).

## 2. Background: SID generation in MiniOneRec
- Prompt → 3 SID tokens (RQ codebooks), trie-constrained beam 50, full-vocab log-softmax,
  forced `\n` + EOS scored. Catalog stats: 3,459 items, 3,444 SIDs, 88 / 2,488 / 3,444 nodes per
  level, max branching 66 / 12.

## 3. Where the time goes (cost model + profile)
- Roofline: 3.1 GB of BF16 weights per forward; A6000 768 GB/s ⇒ ≈4–4.5 ms/forward floor;
  M=50 rows is far below the ≈200-row ridge.
- Profile of the uncompiled engine: GEMM ≈24 ms (65% BW), ≈4,000 elementwise kernels ≈14 ms,
  LM head ≈3–4 ms.

## 4. The engine (speedrun ladder)
| rung | median ms (A6000, batch 1) | fidelity gate |
|---|---:|---|
| HF generate, FP32 | 869 | reference |
| RapidTinyRec FP32 exact stack | 114.8 | bit-exact ranks/scores |
| RapidTinyRec BF16 | 56.5 | HR@10 = |
| speedrun engine, one graph/request, BF16 | 41.3 | top-1 = FP32 for 94% |
| + torch.compile fusion | 35.2 | |
| + Triton shared-prefix beam attention | [TBD] | |
| + padded LM head | [TBD] | |
| + INT8 weights | [TBD] | |
| exact search, no suffix (N=64) | 31.5 (uncompiled) | top-10 certified 83% |
| best combined | [TBD] | |

## 5. Certified exact decoding
- Algorithm (bound U2, U3; certificate).
- Results: certification rates vs N (full / no-suffix / trie); latency vs beam.

## 6. What does exact search reveal? Scoring semantics
- Beam search error under full-vocab objective (search recall@10 vs exhaustive).
- Search-error-helps: N=16 > 32 > 64 > 128 on HR@10 (paired bootstrap).
- Scoring rules × search (offline simulation over exhaustive dumps of all 4,866 users).

## 7. Precision and quantization frontier
- FP32 / FP16 / BF16 / INT8 / INT4: latency vs rank fidelity vs HR/NDCG.

## 8. Throughput
- users/s vs batch on one A6000.

## 9. Related work (from `lit_ledger.md`)

## 10. Limitations
- One dataset/checkpoint (Office); catalog small (exact search scales with #prefixes);
  A6000 only (+L40S); no online serving.
