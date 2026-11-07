# Beam Search Is Optional: Certified Exact Decoding for LLM Generative Recommenders

*Working draft, 2026-11-07. Every number cites a run under `runs/` (A6000 unless noted).
Sections marked [TBD] wait on runs that are queued.*

## Abstract

LLM-based generative recommenders emit an item as a short semantic-ID (SID) token sequence
under a catalog trie and serve it with constrained beam search. We study inference for an
open, widely reproduced instance — MiniOneRec (Qwen2-1.5B, three-level RQ semantic IDs, beam
50) on its two released datasets — and show that beam search is unnecessary. At batch 1, cost is
set by the number of sequential forwards times weight bytes: on an RTX A6000 one forward costs
8.1 ms for 1 token and 9.8 ms for 50, so beam width is nearly free and depth of the decode loop is
everything. Because every scored term is a log-probability ≤ 0, scoring the prompt together with
*all* first-level trie nodes in one forward (trie-level prefill fusion) bounds every catalog item;
one more forward over the best-bounded prefixes yields exact scores and a per-request
certificate that the returned top-K is the global optimum of the model's own objective. With a
hand-written engine (one CUDA graph per request, Triton shared-prefix attention, a cached
prompt template), certified decoding takes **14.4 ms** per request on Office and **14.0 ms** on
Industrial (11.5 ms on an L40S), versus 869 ms for Hugging Face `generate` (60×) and 56.5 ms for the best prior
implementation, and serves 172 users/s on one GPU — 1.9× a fully fused beam-50 engine.
Certification holds for ≥99.7% of users at top-10 under codebook or trie normalization, and was
checked against brute-force scoring of every item (0 violations). Exact search also works as an
instrument: with exhaustive FP32 scores for every (user, item) pair we find that beam-50's
search error under the training objective (it recovers 86% of the exact top-10) does not hurt
quality, that the deep SID terms carry a structural penalty (−1.6 nats for sharing a prefix with
one sibling; ~47% of mass leaked off the codebook), and that the decode-time scoring rule should
be chosen on validation data: on Office a codebook softmax with the third code down-weighted
improves HR@10 by +1.11 points [+0.53, +1.69] (+7%) and NDCG@10 by +0.62 [+0.32, +0.93] over the
FP32 reference decode, while on Industrial no rule beats the original and certified decoding
matches its quality at 1.8× the speed of a fully optimized full-semantics beam engine. FP16 matches BF16's speed with ~10× better fidelity;
weight-only INT8/INT4 do not help at 50–200 rows per forward.

## 1. Introduction

Generative retrieval with semantic IDs is deployed at scale (OneRec, GR4AD, GRACE, PinRec,
Spotify, NVIDIA SID-GR) and serving cost gates its use. Public engines report throughput against
SGLang/vLLM/TensorRT-LLM with *approximate* parity (FlashRec: 86–92% beam-set overlap; NVIDIA:
top-K overlap 0.96) and make silent, different scoring choices (full-vocabulary log-softmax then
mask; per-codebook softmax; mask-before-softmax; `ignore_eos`). We ask, on one fully reproducible
model family with two datasets:

1. Where does the time go at batch 1, and what is the floor?
2. Is beam search necessary when the catalog trie is small?
3. Which score is being maximized, and does maximizing it exactly help?

Contributions:
- **Cost model and measured ladder** from HF `generate` (869 ms) to 14.4 ms on one A6000, with a
  fidelity gate at every rung (FP32 rank agreement; paired-bootstrap HR/NDCG on all test users).
- **Trie-level prefill fusion**: prompt + all first-level trie nodes in one forward; exact;
  removes a sequential step from beam and exact decoding alike.
- **Certified exact top-K decoding** in two forwards (three with a forced suffix), with a
  per-request optimality certificate computed on device; validated against brute force.
- **Exhaustive score dumps** (FP32, every test user × every SID, both datasets) and an offline
  simulator for any search × scoring rule; a scoring-semantics study showing the rule is a
  first-order, dataset-dependent choice; a mechanism for why deep SID terms mislead.
- Negative results: weight-only INT8/INT4 and whole-layer compilation do not help at these
  shapes; one-forward certification is rare (0.4% at top-10); SID codes are decided in the last
  ~6–9 of 28 layers (logit lens), and exiting the third-code forward early keeps HR@10 flat
  but costs HR@1 (9.4 ms at −3.6 pt HR@1 with 6 layers): the third code orders siblings.

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

- **Cached prompt template**: all MiniOneRec prompts share a 69-token instruction prefix; its K/V
  are computed once, so the fused first forward carries only the user-specific tail plus the
  first-level trie nodes (bit-identical outputs in FP32).

Ladder (`runs/ladder2`: one allocation, one RTX A6000, 2 × 500 requests per rung; batch 1; median /
p95 ms; HF and prior rows from earlier RapidTinyRec measurements on the same GPU type and cohort):

| rung | median | p95 | fidelity / quality |
|---|---:|---:|---|
| HF `generate`, FP32 | 868.8 | — | reference |
| RapidTinyRec FP32 bit-exact stack (prior) | 114.8 | 119.9 | bit-exact |
| RapidTinyRec BF16 (prior) | 56.5 | 57.6 | HR@10 = reference |
| speedrun engine, FP32 | 100.8 | 105.5 | top-10 set = reference for 500/500 |
| BF16, one CUDA graph per request (fused suffix) | 42.1 | 42.9 | top-1 = FP32 471/500 |
| + torch.compile | 35.0 | 35.4 | |
| + trie-level prefill fusion + Triton beam attention | 28.1 | 28.4 | HR@10 = FP32 (0.1589) |
| + FP16 + cached template | 26.2 | 27.7 | top-1 = FP32 500/500 |
| certified exact, trie scoring | **14.4** | 15.7 | top-50 certified 99.9%; HR@10 0.1648 |
| certified exact, codebook, w_c = 0.5 (tuned on validation) | **14.4** | 15.6 | top-10 certified 99.7%; HR@10 0.1700 |

The separate-suffix variant (two one-token forwards for `\n` and EOS) costs 55.9 vs 46.3 ms in the
earlier paired ladder (`runs/ladderZ`), i.e. forced-suffix fusion saves ≈10 ms. Allocations of the
same GPU type differ by up to ≈10% in absolute latency; ratios are taken within one allocation.

## 5. Certified exact decoding

```
input : prompt x, catalog trie (first codes A, prefixes (a,b), items (a,b,c)), budgets M, N, K
F1    : one forward over [x ; every a ∈ A as a tree node]          # trie-level prefill fusion
        s(a) for all a ;  s(b|a) for all valid (a,b)
        U2(a,b) = s(a) + s(b|a)                                        # admissible: score ≤ U2
F2    : one forward over the M prefixes with largest U2, token b each, reusing the a-node K/V
        s(c|ab) for all children ;  U3(a,b,c) = U2 + s(c|ab)            # exact if no forced suffix
F3    : (forced suffix only) one forward over the N items with largest U3, tokens [c, \n]
        exact(a,b,c) = U3 + s(\n|abc) + s(EOS|abc\n)
return: items sorted by exact score, and for each K:
        certified_K = exact_(K) ≥ max( U3 of best un-suffixed item , U2 of best unexpanded prefix )
```
Every step has a fixed shape (|A|, M, N), so the whole request is one CUDA-graph replay; the
certificate is computed on the device and returned with the list.

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
  between lp(c|ab) and sibling count over each user's top-50) — steeper than the −log k of an
  even split (−1.6 nats from one to two siblings vs log 2 = 0.69). This is a structural penalty
  on items that share a prefix, unrelated to the user (Fig. `sibling_penalty_office`).
- **Codebook leakage.** Even after a valid (a,b), a median 47% of next-token mass falls outside
  the c-codebook (median log-mass −0.63, flat across sibling counts); full-vocabulary scoring
  charges every item for it, unevenly across users and prefixes. Codebook/trie renormalization removes it.
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
— outside the M ≤ 16 regime where weight-only kernels shine. The roofline makes this explicit: one
A6000 forward costs 8.1 / 9.8 / 12.2 / 16.1 ms at 1 / 50 / 128 / 256 tokens in BF16, versus 5.2 / 8.3 /
11.8 / 18.1 ms with INT8 weights. On an L40S, dynamic per-token FP8 W8A8 (`torch._scaled_mm`,
rowwise scales) is *slower* than BF16 end to end (certified exact 15.9 vs 14.0 ms; beam 23.5 vs
20.1 ms) at near-equal quality, because quantizing activations at every GEMM outweighs the
narrower weights at 50–216 rows; a quantize-in-RMSNorm FP8 path is future work.

## 8. Throughput

Users per second on one A6000 (BF16/FP16, cached template), batch B users per graph replay:

| decoder | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|---:|
| beam-50, full MiniOneRec scoring | 37.5 | 69.2 | 77.0 | 86.0 | 91.3 |
| certified exact, trie scoring | 67.8 | 135.6 | 145.9 | 165.2 | 172.2 |
| certified exact, tuned codebook | 67.8 | — | 160.5 | — | 176.2 |

Exact decoding processes fewer tokens per user (prompt tail + 88 trie nodes + 64 prefix rows vs
prompt + 88 + 50 + 100 beam rows) and stays ≈1.9× ahead once both are compute-bound.

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
