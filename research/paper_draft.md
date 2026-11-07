# Beam Search Is Optional: Certified Exact Decoding for LLM Generative Recommenders

*Working draft, 2026-11-07. Every number cites a run under `runs/` (A6000 unless noted).
Sections marked [TBD] wait on runs that are queued.*

## Abstract

LLM-based generative recommenders emit an item as a short semantic-ID (SID) token sequence
under a catalog trie, and serve it with constrained beam search. We study the inference of
an open, widely reproduced instance — MiniOneRec (Qwen2-1.5B, three-level RQ semantic IDs,
beam 50) — and show that beam search is neither necessary nor the best choice. (1) At batch 1
latency is set by the number of *sequential* forwards times weight bytes, almost independent
of how many tokens each forward processes up to ~200; beam search spends most of its time
re-reading weights to walk a tree whose shape is known in advance. (2) Because the trie is
tiny (88 first codes and 2,488 two-code prefixes for 3,459 Office items) and every scored term
is a log-probability ≤ 0, two forwards suffice to score *every* first code and *every* prefix,
which yields admissible upper bounds on every item; a third, small set of teacher-forced paths
gives exact scores for the best-bounded items, and a simple test certifies when the returned
top-K is the global optimum of the model's own objective. Fusing the first trie level into the
prompt forward removes another sequential step. (3) Exact decoding exposes the objective:
MiniOneRec's full-vocabulary score, including the forced newline/EOS tokens that production
engines silently drop or renormalize away, is mis-calibrated — admitting more candidates
toward its true argmax lowers HR@10, while trie-renormalized scoring is both better and
almost always certifiable. (4) With a hand-written engine (one CUDA graph per request, fused
first level, Triton shared-prefix attention), exact trie-scored decoding runs in **16.1 ms**
per request on one RTX A6000, versus 869 ms for Hugging Face `generate` (54×) and 56.5 ms for
the best prior BF16 implementation, while certifying the top-50 for 99.9% of users and
improving HR@10 from 0.1589 to 0.1646 and NDCG@10 from 0.1247 to 0.1289 over the FP32 reference.

## 1. Introduction

Generative retrieval with semantic IDs is now deployed at scale (OneRec, GR4AD, GRACE,
PinRec, Spotify, NVIDIA SID-GR) and serving cost is the gating factor. Public engines report
throughput against SGLang/vLLM/TensorRT-LLM with *approximate* parity (FlashRec: 86–92%
beam-set overlap; NVIDIA: top-K overlap 0.96) and make silent, different scoring choices
(full-vocabulary log-softmax then mask; per-codebook restricted softmax; mask-before-softmax;
`ignore_eos` fixed-length output). We ask three questions on one fully reproducible
model/dataset pair:

1. Where does the time go at batch 1, and what is the floor?
2. Is beam search necessary when the catalog trie is small?
3. Which score is being maximized, and does maximizing it exactly help?

Contributions:
- **A cost model and a measured ladder** from HF `generate` (869 ms) to 16.1 ms, with a
  fidelity gate at every rung (FP32 rank-exactness for the exact track; paired-bootstrap
  HR/NDCG for precision and semantics changes).
- **Trie-level prefill fusion**: the prompt and all first-level trie nodes in one forward
  (tree attention mask), exact, removing one sequential step from beam *and* exact decoding.
- **Certified exact top-K decoding** for LLM-based SID recommenders in 2 forwards (no forced
  suffix) or 2–3 forwards (with it), with a per-request optimality certificate.
- **A scoring-semantics study**: full-vocab vs codebook vs trie normalization, with and without
  forced suffix tokens, under beam and exact search; the forced suffix costs a forward and
  hurts ranking; trie scoring + exact search is best.
- Replication on a second MiniOneRec dataset (Industrial & Scientific) [TBD], an exhaustive
  FP32 per-token score dump for every test user × catalog item (both datasets) [TBD], and a
  weight-only/FP8 quantization frontier [TBD].

## 2. Setting

**Model and data.** MiniOneRec Office checkpoint (Qwen2-1.5B: 28 layers, d=1536, 12/2 GQA
heads, tied 152,265-row embedding incl. 600 SID tokens). Items: 3,459 (3,444 unique SIDs;
15 two-item collisions) with a 3-level code ⟨a,b,c⟩. Trie: 88 first codes, 2,488 two-code
prefixes, max branching 66 / 12. Test users: 4,866; prompts 76–121 tokens. Latency cohort:
the frozen 500 distinct users of prior RapidTinyRec work.

**Released decoding.** Constrained beam search, width 50; each step adds the full-vocabulary
log-softmax of the allowed token; after `c` the trie forces `\n` then EOS and both are scored;
length penalty 0. Score(item) = lp(a)+lp(b|a)+lp(c|ab)+lp(\n|abc)+lp(EOS|abc\n).

**Metrics.** SID-level HR@K and NDCG@K on all 4,866 test users; paired bootstrap (2,000
resamples over users) for every quality delta. Latency: batch 1, CUDA-synchronized, one graph
replay per request; setup/compilation/capture excluded; median and p95 over 500 users.

## 3. Cost model

A forward reads ~3.1 GB of BF16 weights (1.31 B non-embedding parameters + the tied 0.47 GB
LM head). At 768 GB/s an A6000 needs ≥4.0 ms per forward regardless of how many query tokens
it carries until arithmetic dominates (~150–200 tokens for 155 dense BF16 TFLOPS). Beam-50
issues 5 sequential forwards (prompt; inputs a, b, c, \n), so its floor is ≈20–23 ms in BF16
and ≈40–45 ms in FP32, and its cost barely depends on the beam width. The levers are
therefore (i) fewer sequential forwards, (ii) fewer bytes per forward, (iii) fewer and fused
kernels per forward. [Roofline measurement: TBD `sweepH/roofline_*`.]

Profile of the first engine version (BF16, eager): ≈24 ms GEMM at ≈65% of peak bandwidth,
≈14 ms in ≈4,000 small elementwise kernels, ≈3–4 ms in the 152k-row LM head.

## 4. Engine

- **Hand-written forward** from safetensors: concatenated QKV and gate/up projections,
  vocabulary padded to 128 rows (aligned cuBLAS kernels), RMSNorm/RoPE as in HF Qwen2; Q is
  pre-scaled so FP16 does not overflow on Qwen2's large QKV biases.
- **Shared-prefix KV layout**: one copy of the prompt K/V per user; ≤4 private K/V slots per
  beam row; reorder only the private slots.
- **Fused forced suffix**: score `\n` and EOS with one two-token forward per beam (exact: each
  final beam has a single admissible continuation, so nothing is pruned). −10 ms.
- **Trie-level prefill fusion** (`prefill_tree`): append the |A| first-level trie nodes to the
  prompt; node i attends to the user's prompt and itself at position L. One forward yields
  lp(a) and lp(b|a) for every a. Rank-identical to the unfused engine (FP32 score Δ ≤ 3e-5).
- **Triton shared-prefix beam attention**: one program per (user, row, KV head) packs all query
  heads of the group and all new tokens of the row against the prompt K/V and the private
  slots with a causal rule; fp32 softmax. 2× closer to FP32 than the PyTorch BF16 path.
- **One CUDA graph per (batch, padded length)** with a shared memory pool; torch.compile fuses
  the per-layer elementwise work.

Ladder (A6000, batch 1, median ms; full MiniOneRec semantics unless noted):

| rung | ms | fidelity |
|---|---:|---|
| HF generate, FP32 | 868.8 | reference |
| RapidTinyRec FP32 bit-exact stack (prior) | 114.8 | bit-exact |
| speedrun engine, FP32 | 98.8 | HR@10 = reference |
| RapidTinyRec BF16 (prior) | 56.5 | HR@10 = |
| speedrun, BF16, one graph | 41.3 | |
| + compile | 35.2 | ΔHR@10 −0.10 [−0.27,+0.06] |
| + fused first level + Triton attention | 27.9 | HR@10 0.1589 (= FP32) |
| certified exact, no forced suffix | 19.8 | top-10 certified 94% |
| certified exact, trie scoring | **16.1** | top-50 certified 99.9% |

## 5. Certified exact decoding

All scored terms are log-probabilities ≤ 0. Define U2(a,b) = s(a)+s(b|a) and
U3(a,b,c) = U2 + s(c|ab). For any item, score ≤ U3 ≤ U2.

1. **F1** (fused prefill): prompt + all first codes → s(a) ∀a, s(b|a) ∀(a,b) ⇒ U2 for all items.
2. **F2**: the M best prefixes by U2, one token (b) each, reusing the a-node K/V → s(c|ab) for
   all their children ⇒ U3 (exact when there is no forced suffix).
3. **F3** (only with the forced suffix): the N best items by U3, tokens [c, \n] → exact scores.
4. **Certificate**: top-K is globally optimal iff the K-th exact score ≥ max(U3 of the best
   un-suffixed item, U2 of the best unexpanded prefix).

Validated against brute-force scoring of all 3,444 SIDs in FP32: 0 violations; top-1 always
optimal. Certification rates: full scoring with suffix, N=64: top-10 26–29%; no suffix:
top-10 94%; trie: top-50 99.9%.

## 6. What exact decoding reveals

- Beam-50 is not exact under MiniOneRec's objective: for some users, half of the exact top-10
  never enter the beam (their (a,b) prefix ranks 100+ by prefix score but their c/suffix terms
  are small). The FP32 beam reference has a non-optimal top-1 for 3/500 cohort users.
- **Search error helps under the full objective.** Admitting fewer candidates by prefix bound
  improves HR@10 monotonically: N=16 +0.70 pt [+0.12,+1.29] vs the FP32 reference; N=32 +0.64;
  N=64 +0.10; N=128 −0.02 (≈ beam).
- **The forced suffix hurts.** Dropping lp(\n)+lp(EOS) improves NDCG@10 by +0.36 pt
  [+0.13,+0.61] and removes a forward; under that objective beam-50 and certified exact search
  give identical HR@10 (0.16235).
- **Trie scoring is best** and is certifiable: HR@10 0.1646, NDCG@10 0.1289. [Paired CIs and the
  full offline search × scoring matrix from the exhaustive dumps: TBD.]

## 7. Precision and quantization [TBD]

BF16 is free on quality (ΔHR@10 −0.10 pt, CI includes 0) though it reorders lists (top-10 set
identical to FP32 for 35% of users). FP16 needs Q pre-scaling and is ≈10× closer to FP32 than
BF16. Weight-only INT8 via GemLite at M≈50 gains only ≈10% uncompiled. [Compiled INT8/INT4,
FP8 on L40S: TBD.]

## 8. Throughput

Compiled BF16 beam: 23.9 users/s (B=1) → 71.5 (B=16) → 77.9 (B=64) on one A6000;
compute-bound beyond B≈8. [Exact decoders: TBD.]

## 9. Related work

See `lit_ledger.md`. Systems lineage: FlashRec, NVIDIA SID-GR, xGR, GRACE, GR4AD, STATIC,
FlashTrie, STARK (tree attention for beams), Hydragen/FlashInfer cascade. Prefill-only heads
(NEZHA, SID-MLP, Deco) are complementary. Search: DiGR's brute-force exhaustive scoring
(5 s/user) and 2611.01072's first-step coverage study for TIGER; Stahlberg & Byrne (2019) on
exact search in NMT. Constrained-decoding theory: Wu et al. (SIGIR'25), grammar-aligned decoding.

## 10. Limitations

One model family; small catalogs (exact decoding scales with the number of first-level codes
and the prefix bound's tightness — the certificate tells you when it failed); A6000 (+L40S)
only; batch-1 latency is the primary metric.
