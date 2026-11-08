# Beam Search Is Optional: Certified, Late-Entry Decoding for LLM Generative Recommenders

*Working draft, 2026-11-08. Every number cites a run under `runs/` (GPU named in each table).
Sections marked [TBD] wait on runs that are queued.*

## Abstract

LLM-based generative recommenders emit an item as a short semantic-ID (SID) token sequence under
a catalog trie and serve it with constrained beam search. On an open, widely reproduced instance —
MiniOneRec (Qwen2-1.5B, three-level SIDs, beam 50) on its two released datasets — we show that
beam search can be replaced by **certified exact decoding**. Every scored term is a
log-probability ≤ 0, so one forward over the prompt plus every first-level trie node bounds every
catalog item; one more forward over the best-bounded prefixes yields exact scores and a
per-request certificate that the returned top-K is the global optimum of the model's objective
(0 violations against brute force). Under trie normalization the bound is nearly tight —
certifying the top K costs about K prefix expansions at every K up to 1,000 — and single-child
prefixes need no second forward at all, so a certified **top-500 still costs two forwards: 6.1 ms
on an H100 vs 18.1 ms for MiniOneRec's beam-500 (3.0×; 6.4× in throughput)**. We further find that
**the third SID code does not need the lower layers**: prefix rows can *enter* the second forward
at layer 14–22 of 28 from a small bridge distilled from their parent node's state (validation
prompts only, ≈30 s, no labels, frozen model). This is statistically lossless on Office from 6
remaining layers and costs ≤0.35 points of HR@1 on Industrial, whereas early exit at the same cost
loses 2.4–3.7 points. With a hand-written engine (one CUDA graph per request, Triton attention for
the fused first forward), certified late-entry decoding serves a request in **3.3 ms on an H100
and 9.7 ms on an RTX A6000** (Hugging Face `generate`: 869 ms on the A6000), and **1,369 users/s**
on one H100. Exact search is also an instrument: the beam-search curse grows with list length —
exact search under the training objective loses 4.3 points of HR@500 on Office — while a
validation-chosen normalized, depth-discounted objective reverses it (+2.4 points HR@500, +1.1
HR@10). FP16 matches BF16's speed with ~10× better fidelity; weight-only INT8/INT4 and FP8 do not
help at these shapes.

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
- **Certified exact top-K decoding** in two forwards (three with a forced suffix), with a
  per-request optimality certificate computed on device; validated against brute force.
- **The certificate's cost scales with K, not with the catalog**: under trie normalization
  certifying the top K expands ≈K prefixes, and only multi-child prefixes need the second forward
  (≈0.4 K rows). Certified top-500 is two forwards; MiniOneRec's beam-500 is three wide ones.
- **Late entry**: deeper SID codes skip the lower layers via a bridge distilled from the parent
  node's state at the entry layer (frozen model, no labels); lossless on Office at 6–14 of 28
  layers, ≤0.35 pt HR@1 on Industrial; composes with the certificate.
- **Engine and ladders** on an A6000 and an H100 with a fidelity gate at every rung: trie-level
  prefill fusion (exact), Triton shared-prefix and prompt attention, cached prompt template,
  one CUDA graph per request. HF 869 ms → 9.7 ms (A6000); FP32 48 ms → 3.3 ms (H100).
- **Exact search as an instrument** (exhaustive FP32 scores for every user × item, both
  datasets): the beam-search curse grows with list length; the decode-time scoring rule is a
  first-order, dataset-dependent choice; the deep SID terms carry a structural sibling penalty.
- Negative results: one forward is not enough (a user-independent third-code table loses 4.8 pt
  HR@1; speculating the expansion set covers ≤29% of requests; scoring all multi-child prefixes in
  one late-entry forward is exact but slower than two forwards); late entry of the *second* code is
  a trade-off; weight-only INT8/INT4 and FP8 W8A8 do not help at 50–700 rows per forward.

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

Two refinements from the final engine. (i) On the A6000 the fused first forward is
*compute*-bound: its GEMMs take 4.84 / 5.78 / 6.58 / 7.19 ms per forward at 16 / 64 / 140 / 216 rows
(cuBLAS; a tuned split-K Triton kernel helps only at ≤64 rows), and at 140 rows the gate/up GEMM
already runs at ≈57 TFLOPS. Fewer rows, not better kernels, is the lever there. (ii) On the H100
the engine is *kernel-count*-bound: the first certified decoder issued 2,555 kernels and 5.9 ms of GPU
time per request, with GEMMs at ≈50% of HBM bandwidth (≈3.1 ms) and ≈2.8 ms of small copy,
concatenation and attention-preparation kernels; streaming the weights twice would take ≈1.6 ms.
Late entry (fewer layers in the second forward) and a Triton prompt attention (fewer kernels in the
first) cut this to 1,474 kernels and 3.7 ms.

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

- **Triton prompt attention for the fused first forward** (`--tree-fast`): the template K/V is
  copied into the cache once per request, prompt rows attend causally over the cache with a
  Triton kernel, trie-node rows reuse the shared-prefix beam kernel; no per-layer concat, K/V-head
  repetition or mask tensors. H100: 5.47 → 4.56 ms for the certified top-10 decoder, identical
  quality.
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

Same ladder on an H100 (`runs/h100M`, one allocation, 2 × 500 requests per rung):

| rung (H100) | median | p95 | fidelity / quality |
|---|---:|---:|---|
| speedrun engine, FP32 | 48.3 | 50.2 | top-10 set = reference 500/500 |
| BF16, one CUDA graph | 20.1 | 22.4 | top-1 = FP32 465/500 |
| + torch.compile | 12.9 | 13.0 | |
| + trie-level prefill fusion + Triton beam attention | 8.50 | 8.66 | top-1 = FP32 491/500 |
| + FP16 + cached template | 8.48 | 8.59 | top-1 = FP32 500/500 |
| certified exact, trie | **5.57** | 5.74 | top-50 certified 100% (cohort) |
| certified exact, codebook w_c = 0.5 | **5.66** | 5.76 | HR@10 0.1697 (full test) |

On the H100 the engine is kernel-bound rather than bandwidth-bound: 2,555 kernels and 5.9 ms
of GPU time per request, of which GEMMs are ≈3.1 ms (≈50% of HBM bandwidth) and small
copy/concat/attention-prep kernels ≈2.8 ms; streaming the weights twice would take ≈1.6 ms.
Compiling whole decode layers (incl. the Triton attention) saves 3.5% here (5.15 vs 5.33 ms in a
paired run) but nothing on the A6000. FP8 W8A8 is slower on the H100 too (6.0 ms).

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

### 5.1 What the certificate costs, as a function of K

The second forward must expand every prefix whose bound exceeds the K-th exact score,
{(a,b) : U2(a,b) > s_K}. From the exhaustive FP32 scores (all test users, both datasets):

| prefixes to expand (median / p90 / p99) | K=10 | K=50 | K=100 | K=500 | K=1000 |
|---|---|---|---|---|---|
| Office, trie | 10 / 12 / 14 | 52 / 57 / 61 | 104 / 111 / 117 | 489 / 510 / 524 | 932 / 960 / 979 |
| Office, codebook w_c=0.5 | 10 / 13 / 35 | 51 / 61 / 180 | 100 / 117 / 325 | 459 / 517 / 1,058 | 871 / 962 / 1,600 |
| Office, full vocabulary (MiniOneRec) | 52 / 1,715 / 2,488 | 268 / 2,462 / 2,488 | 506 / 2,486 / 2,488 | — | — |
| Industrial, trie | 10 / 13 / 16 | 52 / 60 / 65 | 103 / 115 / 123 | 470 / 501 / 522 | 881 / 922 / 948 |

Under trie normalization the bound is almost tight: certifying the top K costs about K prefix
expansions at every K, with a thin tail. Under the full-vocabulary objective the off-codebook
leak and the suffix make U2 loose and certification needs nearly the whole trie. The
normalization choice therefore decides whether exact search is affordable at all.

Trie normalization gives a second saving. The third-code term is a log-softmax over the valid
children of (a,b), so it is exactly 0 when (a,b) has one child: every item under a single-child
prefix is exactly scored by the first forward. Only multi-child prefixes need the second
forward (Office: 581 of 2,488 prefixes, holding 1,537 of 3,444 items):

| multi-child prefixes to expand (median / p90 / p99) | K=10 | K=50 | K=100 | K=500 |
|---|---|---|---|---|
| Office, trie | 5 / 8 / 10 | 22 / 31 / 38 | 43 / 55 / 64 | 175 / 198 / 213 |
| Industrial, trie | 6 / 9 / 12 | 29 / 37 / 43 | 53 / 65 / 75 | 205 / 238 / 259 |

`ExactTrieEngine` expands only multi-child prefixes; its candidate pool is the expanded children
plus every single-child item, both exact. It reproduces the three-stage engine rank-for-rank and
score-for-score (FP32). A certified top-500 is still two forwards, the second carrying ≈210
rows; beam-500 needs four forwards of 500 rows each (three with trie scoring).

**One forward is not enough.** Two ways to drop the second forward fail:
- *A user-independent third-code table* (per-item mean of s(c|ab) over validation users) keeps
  HR@10 within 0.9 pt but drops HR@1 from 0.098 to 0.050 (−4.8 pt [−5.5, −4.1]). Item identity
  explains 77% of the variance of s(c|ab) overall but only 42% within a user's top-50; the third
  code orders siblings per user.
- *Speculating the expansion set before the first forward* (from the user's previous request,
  history prefixes or popularity) covers the full certificate set for top-10 in at most 29% of
  requests with 128 guessed prefixes; certification needs the near-boundary prefixes too.
Certification after the first forward alone happens for 44% of users at K=1 and 0.5% at K=10.

### 5.2 Late entry: the third code does not need the lower layers

The logit lens shows SID codes are decided in the last 6–9 of 28 layers, and early exit (running
only the *first* k layers for the third code) keeps HR@10 but loses 2.4–3.7 points of HR@1. We
instead let the prefix rows of the second forward *enter late*: a row (a,b) starts at layer e with
an input built from its parent node's residual stream at layer e — already computed for every
first code in forward 1 — and runs layers e..27 with exact attention to the prompt and parent
K/V. The input comes from a small bridge,

    x_e(a,b) = h_e(a) + R_e [h_e(a); E[b]; E[a]; 1] + MLP_e([rms(h_e(a)); E[b]; E[a]]),

where R_e is a ridge fit (label-free) and the MLP (2,048 hidden units, output zero-initialized)
is distilled from the model itself: KL between the full-depth and the late-entry third-code
distributions on MiniOneRec *validation* prompts, weighting each prefix row half uniformly and
half by the softmax of its bound U2 (the rows that decide the ranking). Training takes 1–2 minutes
on one H100; no item labels are used and the base model is frozen. The certificate holds for the
late-entry objective. At serving time the bridge runs in FP16 (two small GEMMs).

All test users, validation-tuned codebook objective (w_c = 0.5), FP16, paired against the
full-depth third code (Office HR@1 0.0980 / HR@10 0.1706 / NDCG@10 0.1312; Industrial 0.0847 /
0.1613 / 0.1181):

| second forward | layers | ΔHR@1 | ΔHR@10 | ΔNDCG@10 | top-1 = full |
|---|---:|---|---|---|---:|
| Office, early exit (first k layers) | 6 | −3.68 [−4.34, −3.08] | −0.84 [−1.29, −0.39] | −2.11 [−2.44, −1.77] | 53% |
| Office, late entry, ridge bridge | 6 | −0.51 [−0.86, −0.14] | −0.58 [−1.05, −0.14] | −0.54 [−0.80, −0.29] | 66% |
| Office, late entry, distilled bridge (uniform KL) | 6 | −0.12 [−0.37, +0.12] | −0.12 [−0.35, +0.08] | −0.08 [−0.22, +0.04] | 92% |
| **Office, late entry, distilled bridge (weighted KL)** | **6** | **+0.27 [+0.12, +0.45]** | **−0.04 [−0.25, +0.16]** | **+0.10 [−0.01, +0.20]** | **94%** |
| Office, distilled bridge (uniform) | 2 / 4 / 10 / 14 | −0.45 / −0.27 / +0.06 / −0.02 | −0.09 / +0.04 / — / +0.14 | −0.24 / −0.09 / −0.00 / +0.05 | 86–94% |
| Industrial, early exit | 10 | −2.38 [−2.96, −1.83] | −0.77 [−1.30, −0.20] | −1.80 [−2.18, −1.40] | 47% |
| Industrial, distilled bridge (uniform KL) | 10 | −0.35 [−0.62, −0.11] | −0.04 [−0.31, +0.20] | −0.18 [−0.33, −0.04] | 86% |
| **Industrial, distilled bridge (weighted KL)** | **10** | **−0.07 [−0.22, +0.09]** | **−0.04 [−0.24, +0.18]** | **−0.04 [−0.13, +0.05]** | **92%** |

With prefix-weighted distillation, late entry is statistically indistinguishable from the
full-depth second forward on HR@10 and NDCG@10 on both datasets (Office at 6 remaining layers,
Industrial at 10), and stays above the MiniOneRec reference decode (Office HR@10 0.1702 vs
0.1589). The small HR@1 gain on Office is a calibration side effect of the bridged objective, not
a claim we rely on. Late entry of the *second* code (the 88 first-code rows joining forward 1 at
layer 22, bridged from the last prompt token) is a trade-off: ΔNDCG@10 −0.25 [−0.51, −0.01], and
only 61% of top-10 prefixes agree with full depth; the second code decides set membership.

Engine latency (CUDA graph, FP16, paired within one allocation; certified top-10):

| | A6000 | H100 |
|---|---:|---:|
| certified exact, trie, M=16 | 13.65 | 5.03–5.47 |
| + Triton prompt attention in forward 1 | 13.26 | 4.56 |
| + late entry at layer 22 | **9.06** | 3.56 |
| + whole-layer compile, FP16 bridge | 9.07 | **3.24** |
| certified exact, tuned codebook, M=64 → + late entry 22 | 14.33 → 9.68 | 5.57 → 4.17 |
| certified top-500 → + late entry (and fast forward 1 on the H100) | 17.95 → [TBD] | 6.06 → 3.81 |

## 6. What exact decoding reveals about the objective

**Beam search error is real but is not what limits quality.** From exhaustive FP32 scores of every
catalog SID for every test user (`scripts/dump_scores.py`), beam-50 under MiniOneRec's
full-vocabulary objective recovers 86% of the exact top-10 (top-1 exact for 99.8% of users), yet
exhaustive search gives the same HR@10. Under codebook or trie normalization beam-50 is
essentially exact (99.6% / 99.9% of the exact top-10).

**The beam-search curse grows with list length.** Generative retrieval feeds hundreds of
candidates to a ranker, so lists longer than 10 matter. Beam-W recovers only part of the exact
top-W under the full objective (Office 49% at W=50, 53% at 100, 64% at 500; Industrial 67 / 69 /
79%); under trie scoring 95 / 96 / 99.6%. Under the full objective, exact search *loses* recall
and the loss grows with K (Office, paired bootstrap vs MiniOneRec's beam-W decode): HR@50 −1.8 pt
[−2.5, −1.1], HR@100 −2.5 [−3.3, −1.7], HR@200 −3.5 [−4.4, −2.5], HR@500 −4.3 [−5.3, −3.2];
Industrial −0.1 to −0.7 (n.s.). With the validation-tuned codebook objective exact search
*gains* recall on both datasets: Office HR@100 +2.3 [+1.5, +3.0], HR@500 +2.4 [+1.4, +3.3];
Industrial HR@200 +0.9 [+0.1, +1.6], HR@500 +0.9 [+0.0, +1.8]. Beam search's prefix-first pruning
acts as an implicit depth discount; once the discount is explicit, exactness stops hurting.

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

H100 (`runs/h100M`, `runs/largeKH`, `runs/tpK`, `runs/lateLH`), users/s:

| decoder | B=1 | B=8 | B=32 | B=64 |
|---|---:|---:|---:|---:|
| certified exact, tuned codebook (M=64) | 178 | 726 | 904 | 920 |
| certified exact, trie, multi-child prefixes only (M=16) | 182 | 837 | 1,223 | — |
| + late entry at layer 22 | 261 | 983 | 1,355 | 1,369 |

Large candidate lists (H100, users/s): certified top-500 173 / 374 / 414 / 450 at B = 1/4/8/16;
beam-500 with trie scoring 132 / 229 / 253 (B = 1/4/8); MiniOneRec's beam-500 (full vocabulary,
forced suffix) 54 / 68 / 71 — **6.4×** at the largest batch each fits. Certified top-100: 185 / 689 /
858 (B = 1/8/32) vs trie beam-100 169 / 628 / 740.

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

**Skipping layers.** Early exit (CALM, Schuster et al., 2022; LayerSkip, Elhoushi et al., 2024)
truncates the *top* of the network; Mixture-of-Depths (Raposo et al., 2024) routes tokens around
blocks with trained routers. SkipDecode (Del Corro et al., 2023) is closest in shape: later tokens
skip *lower* layers, with the skipped layers' K/V copied from the last computed layer, after
fine-tuning the model for that schedule. EAGLE (Li et al., 2024) drafts next-token *features* from
the target model's top-layer feature and the next token's embedding. Late entry differs from all
of these: the base model is frozen, the entry state of a child row is predicted from its *parent's*
state at the same depth (which the trie-level prefill already computes), and the third-code
forward that it shortens sits inside a certified search, so the speed-up composes with exactness
guarantees for the (bridged) objective. We find the third SID code needs only its last 6–14 of 28
layers when entered this way, whereas the second code is more sensitive.

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

- One model family (MiniOneRec, Qwen2-1.5B) and small catalogs (≈3.5k items, 48–88 first codes).
  Exact decoding's first forward grows with the number of first codes (a 256-code first level is
  still one forward of ≈300 rows); the second forward grows with ≈0.4 K multi-child prefixes; the
  certificate reports when the bound was not tight enough. The one-pass variant (all multi-child
  prefixes in the first forward) only makes sense for small catalogs.
- Late entry adds a learned component (≈10 M parameters per entry layer). The certificate then
  certifies the top-K of the *bridged* objective; we report quality against the full-depth
  objective and the MiniOneRec reference decode on all test users, and train bridges on the
  validation split only. Bridges are per model and per entry layer.
- Batch-1 latency on two GPU types (RTX A6000, H100; L40S for FP8); throughput up to batch 64.
  Absolute latency varies by up to ≈10% between allocations of the same GPU type; all
  comparisons we draw conclusions from are paired within one allocation.
- We did not run FlashRec or NVIDIA's SID-GR engine on matched hardware; our comparison points are
  Hugging Face `generate`, the prior RapidTinyRec stacks and our own beam engine.
