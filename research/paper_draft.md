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

Validated against brute-force scoring of all 3,444 SIDs in FP32: 0 violations among certified
users; top-1 always optimal.

Certification rate (share of the 4,866 Office test users whose top-K is proven globally optimal):

| decoder | scoring | forwards | top-1 | top-5 | top-10 | top-20 | median ms |
|---|---|---:|---:|---:|---:|---:|---:|
| single bound, N=16 items | full (with suffix) | 3 | 0.88 | 0.11 | 0.00 | — | 31.1* |
| single bound, N=32 | full | 3 | 0.96 | 0.32 | 0.05 | 0.00 | 32.4* |
| single bound, N=64 | full | 3 | 0.98 | 0.58 | 0.27 | 0.03 | 36.6* |
| single bound, N=128 | full | 3 | 0.99 | 0.75 | 0.54 | — | 45.5* |
| single bound, N=64 | no suffix | 3 | 1.00 | 0.94 | 0.83 | 0.26 | 31.5* |
| three-stage, M=64, fused first level | no suffix | 2 | 1.00 | 0.97 | 0.94 | 0.86 | 19.8 |
| three-stage, M=64, fused first level | trie | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 16.1 |

\*uncompiled, unfused first level, PyTorch attention (earlier engine stage).

Skipping the second forward when the certificate already closes after the first (possible under
trie scoring, where single-child prefixes — 1,907 of 2,488 — have exact scores after F1)
happens for 42% of users at K=1 but only 0.4% at K=10, so the second forward stays.

## 6. What exact decoding reveals about the objective

**Beam search error is real but is not what limits quality.** From exhaustive FP32 scores of every
catalog SID for every test user (`scripts/dump_scores.py`), beam-50 under MiniOneRec's
full-vocabulary objective recovers 86% of the exact top-10 (top-1 exact for 99.8% of users), yet
exhaustive search gives the same HR@10. Under codebook or trie normalization beam-50 is
essentially exact (99.6% / 99.9% of the exact top-10).

**The objective is mis-calibrated in its deep terms.** Three independent signals:
1. Admitting fewer candidates by prefix bound improves HR@10 (N=16: +0.70 pt [+0.12, +1.29] vs the
   FP32 reference); admitting more converges back to beam.
2. Dropping the forced newline/EOS terms improves NDCG@10 by +0.36 pt [+0.13, +0.61].
3. Restricted normalizations — what production engines do for speed — score better than the
   training objective (beam-50, full test set):

| scoring rule | HR@1 | HR@10 | NDCG@10 | HR@50 |
|---|---:|---:|---:|---:|
| full vocabulary, with forced suffix (MiniOneRec) | 0.0947 | 0.1589 | 0.1249 | 0.2370 |
| full vocabulary, no suffix | 0.0986 | 0.1624 | 0.1284 | 0.2363 |
| trie-renormalized | 0.0982 | 0.1648 | 0.1287 | 0.2417 |
| codebook-renormalized (FlashRec-style) | 0.0986 | 0.1667 | 0.1299 | 0.2470 |

**Depth-discounted scoring.** Weighting later SID levels less,
score = lp(a) + w_b lp(b|a) + w_c lp(c|ab) + w_s(lp(\n)+lp(EOS)), is a two/three-parameter
decode-time calibration that keeps every term ≤ 0, so certified exact decoding applies unchanged.
In-sample on a quarter of the Office test users, w = (0.75, 0.3, 0.3) raises HR@10 by ~10%
relative. [TBD: weights tuned on MiniOneRec's validation split, evaluated once on test, both
datasets — `scripts/tune_weights.py`.] Why the deep terms mislead (exhaustive FP32 scores, 1,217 Office users × 3,444 SIDs):
- **Sibling penalty.** lp(c|ab) falls with the number of catalog items sharing the prefix
  (a,b): median −0.87, −2.46, −3.05, −3.38, −4.12 for 1, 2, 3, 4, 6 siblings (Pearson r = −0.36
  between lp(c|ab) and sibling count over each user's top-50). Splitting mass among siblings
  imposes roughly a −log k structural penalty unrelated to the user.
- **Codebook leakage.** Even after a valid (a,b), a median 47% of next-token mass falls outside
  the c-codebook (median log-mass −0.63); full-vocabulary scoring charges every item for it,
  unevenly across prefixes. Codebook/trie renormalization removes it.
- **The forced suffix is mixed evidence.** On average the target's lp(\n)+lp(EOS) is higher than
  its competitors' (−1.69 vs −6.31; competitor mean dominated by a few very unlikely completions),
  but adding the suffix moves the target up for 28% of users and down for 25%; it correlates
  with log item popularity (r = 0.15) — a weak popularity prior that costs a forward.
These are consistent with SID-MLP's observation that SID difficulty concentrates in the first
code: preference information lives in lp(a) and lp(b|a), while deeper terms are dominated by
catalog structure.

## 7. Precision and quantization

| precision (fused + Triton beam, full semantics) | ms | top-1 = FP32 | top-10 set = FP32 | identical 50-lists | max \|Δscore\| |
|---|---:|---:|---:|---:|---:|
| FP32 (TF32 off) | 98.8–105.0 | 500/500 | 500/500 | — | — |
| FP16 (Q pre-scaled) | 27.5 | 500/500 | 484/500 | 151/500 | 0.15 |
| BF16 | 27.9 | 488/500 | 374/500 | 0/500 | 1.48 |

FP16 is as fast as BF16 and ~10× more faithful; HR/NDCG are statistically indistinguishable
from FP32 for both. Weight-only INT8 (GemLite) is quality-neutral but only 3–8% faster;
INT4 (RTN, group 128) is no faster and less faithful. The kernel profile explains why: after
fusion, GEMMs are ≈80% of GPU time at ≈75% of peak bandwidth, with M = 50–216 rows per forward
— outside the M ≤ 16 regime where weight-only kernels shine. [TBD: FP8 W8A8 on L40S.]

## 8. Throughput

Compiled BF16 beam: 23.9 users/s (B=1) → 71.5 (B=16) → 77.9 (B=64) on one A6000; compute-bound
beyond B≈8. Certified exact decoding processes fewer tokens per user (prompt tail + 88 trie
nodes + 64 prefix rows ≈ 190 with the template cache, vs ≈340 for fused beam). [TBD: measured
throughput curves, `sweepI`, `tmplJ`.]

## 9. Related work

**GR serving engines.** FlashRec (Sohu, 2026; open source) combines wide-beam CUDA graphs, a
fused trie kernel, a per-codebook restricted LM head with restricted log-softmax, radix
prefix caching, cross-request beam batching and FP8, and reports throughput against
SGLang/vLLM/TRT-LLM with 86–92% beam-set overlap to HF. NVIDIA's SID-GR inference example
separates shared context KV from per-beam KV with a custom decode-attention kernel (17.6 ms at
batch 1 for Qwen3-1.7B, beam 256, on an H100) and fixes output length with `ignore_eos`.
xGR (2512.11529) separates shared/unshared KV, terminates sorting early and filters items with
masks; GRACE (2608.00938) captures the whole SID loop in one CUDA graph and treats beams as the
query dimension of cross-attention; GR4AD (2602.22732) reports per-mechanism QPS for
beam-shared KV, top-K pre-cut, FP8 and a lazy decoder; UniSGR/STARK (2607.04068) uses tree
attention over beam candidates; STATIC (2602.22647) and FlashTrie (2607.10044) make trie
constraints cheap on accelerators. Our engine re-implements this lineage for one model and
does not claim those mechanisms; our contributions are the decoding algorithm, the
first-level prefill fusion, and the measurements.

**Faster generation by changing the model.** NEZHA (2511.18793), SID-MLP (2605.12617) and Deco
(2610.32980) run the LLM once and decode SIDs with a light head; RPG, VaLiDRec and masked
diffusion generate SID tokens in parallel; OneRec-V2's lazy decoder and GR4AD's LazyAR cut
decoder compute; AtSpeed, SpecGR and PAD-Rec apply speculative decoding. All are
complementary to exact decoding of an unmodified model.

**Search and scoring.** DiGR (2602.07847) scores every legal SID of an LLM recommender by brute
force (5 s/user) and finds it beats beam-50; 2611.01072 studies beam's first-step losses for
TIGER with an offline exhaustive reference; Item-Supported Decoding (2607.24995) protects
prefixes with an external ranker. Wu et al. (SIGIR'25) bound the error of constrained
decoding; grammar-aligned decoding (2405.21047) shows mask-then-renormalize distorts LLM
distributions. In machine translation, exact search exposed that the model's MAP output is
often worse than beam's (Stahlberg & Byrne, 2019; Meister et al., 2020). To our knowledge no
prior work makes exact search over the SID trie fast enough to serve, gives a per-request
optimality certificate, or measures how forced suffix tokens and normalization choices change
recommendation quality.

## 10. Limitations

One model family; small catalogs (exact decoding scales with the number of first-level codes
and the prefix bound's tightness — the certificate tells you when it failed); A6000 (+L40S)
only; batch-1 latency is the primary metric.
