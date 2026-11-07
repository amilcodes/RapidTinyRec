# Research plan: how fast can LLM-based generative recommendation go?

Working title: **"Beam search is optional: certified exact and roofline-bound inference for
LLM-based generative recommenders."**

## What the literature review says (Nov 2026)

See `lit_ledger.md`, `architecture_directions.md`, `kernel_techniques.md`.

- Every *systems* mechanism we used to go 869 → 115 → 57 ms already has 2025–26
  precedent: shared prefix KV (xGR, NVIDIA SID-GR, GR4AD, GRACE), GPU trie tables (STATIC,
  FlashTrie, FlashRec), whole-loop CUDA graphs (GRACE, FlashRec), restricted LM head
  (FlashRec), FP8 (OneRec-V2, GR4AD). These are lineage, not contributions.
- Prefill-only / distilled SID heads are taken (NEZHA WWW'26, SID-MLP 2605.12617, Deco
  2610.32980).
- **Open**:
  1. *Certified exact top-K search* for LLM-based SID GR. DiGR (2602.07847) showed
     brute-force exhaustive scoring beats beam-50 on NDCG but needed 5 s/user; 2611.01072
     measured beam's first-step losses for TIGER offline. Nobody makes exact search *fast*.
  2. *Scoring semantics*: engines disagree (full-vocab+mask, codebook-restricted
     log-softmax, mask-before-softmax, `ignore_eos` dropping the suffix) and nobody has
     measured the ranking effect.
  3. *Weight-only INT8/INT4* effect on GR ranking (only FP8 W8A8 studied).
  4. A reproducible *batch-1 ladder* with a correctness gate at every rung.

## Contributions (target)

1. **Cost model.** At batch 1 an LLM recommender is weight-read-bound: latency ≈
   (#sequential forwards) × (weight bytes) / BW, nearly independent of beam width up to
   ~150–200 rows. So the levers are forward count, bytes per forward, and kernel overhead.
2. **Speedrun engine.** Hand-written Qwen2 forward + device-resident constrained beam +
   fused forced-suffix forward, one CUDA graph per request. Ladder from HF (869 ms) with
   precision-separated fidelity gates (FP32 rank-exact vs historical reference; BF16/FP16 by
   overlap + HR/NDCG).
3. **Certified exact search ("beam-free GR").** All log-probs ≤ 0 ⇒ lp(a)+lp(b|a) is an
   admissible bound on any item under (a,b). Forward 1 = prefill → lp(a); forward 2 = all
   88 valid first codes as tree nodes → lp(b|a) for all 2,488 prefixes; forward 3 =
   teacher-force [b,c,\n] for the N best-bounded items → exact five-term scores. If the
   K-th exact score ≥ best unexpanded bound, the top-K is provably the global optimum of the
   model's own objective. Same or fewer forwards than beam-50, no search error.
4. **Search error measurement.** Beam-50 vs exhaustive exact on 500 users (search recall@K,
   quality of the exact optimum vs beam's).
5. **Scoring-semantics study.** Full-vocab vs codebook-restricted vs trie-renormalized ×
   with/without forced suffix; effect on HR/NDCG and on #forwards.
6. **Precision/quantization frontier.** FP32 / TF32 / FP16 / BF16 / W8A16 / W4A16 (and FP8
   on L40S): latency vs rank fidelity vs HR/NDCG.
7. **Throughput.** Users/s vs batch size on one A6000; the batch>4 regime turns
   compute-bound.

## Status log

- 2026-11-07 00:20 — Speedrun engine (eager) reproduces FP32 reference 50/50 ranks on MPS
  (scores within 4e-5), batched + padded too. FP16 needs q pre-scaling (Qwen bias overflow);
  then FP16 is ~20× closer to FP32 than BF16 (max score error 0.05 vs 0.9).
- 2026-11-07 00:25 — A6000 smoke, BF16, CUDA graph, no compile: **41.3 ms median / 41.9 p95**
  (prior BF16 stack 56.5). Full Office test (4,866): HR@10 0.158, NDCG@10 0.124.
  Throughput 23.9 users/s (B=1), 57.7 (B=8).
- 01:00 — Kernel profile (BF16, eager, per request): ~24 ms GEMM (≈65% of A6000 peak BW),
  ~14 ms in ~4,000 tiny elementwise kernels (RMSNorm/RoPE/attention glue), ~3–4 ms full-vocab
  LM head (152k rows, 3 calls). ⇒ levers: fusion (compile + Triton beam attention), weight
  bytes (INT8/INT4), restricted LM head.
- 01:00 — Exact search (BF16, single-bound variant, N=32 items): **32.4 ms** vs beam 41.3 ms
  (3 forwards vs 4); full test HR@10 **0.165 vs 0.158**, NDCG@10 0.127 vs 0.124. Top-1
  certified for 96% of users, top-10 for 5% (U2 bound is loose under full-vocab scoring).
  BF16 noise (max score error vs FP32 ≈3.6) makes BF16 certification/exhaustive comparisons
  unreliable → science runs moved to FP32.
- 01:10 — Local FP32: under trie-renormalized scoring the 3-stage exact search certifies
  top-20 for every test user; under full-vocab scoring certification mostly stops at
  top-1/5 because item scores fall off steeply (e.g. −0.04, −11.8, −24.4) and beam missed a
  better 3rd item (−24.4 vs beam −27.2) — real beam search error.
- 01:15 — torch.compile (qkv/post fused) BF16 beam: **35.2 ms** (from 41.3).
- 01:15 — **Search-error-helps signal.** BF16 exact search, full-vocab scoring, paired bootstrap
  over 4,866 users: N=32 HR@10 0.1652 vs N=64 0.1599 (Δ −0.53 pt [−0.95, −0.14]) vs
  N=128 0.1587 (Δ −0.66 [−1.11, −0.23]); NDCG@10 likewise. Items are admitted by the prefix
  bound lp(a)+lp(b|a) and ranked by the full five-term score; admitting more items (closer
  to the true argmax) *hurts*. Hypothesis: the c-level and forced \n/EOS terms of the
  full-vocabulary objective are mis-calibrated for ranking (cf. Stahlberg & Byrne 2019 for NMT).
  Test: FP32 sweep over scoring rules (full / no-suffix / codebook / trie) × (beam / exact).
- 01:25 — **Certificate validated in FP32** (exact N=64 vs brute-force scoring of all 3,444
  SIDs, 100 users): 0 violations among 23 certified-top-10 users; top-1 = global optimum for
  100%. The FP32 beam-50 reference returns a non-optimal top-1 for 3/500 cohort users.
  FP32 full-test quality matches BF16 (HR@10 0.1595 vs 0.1599 at N=64).
- 01:25 — Throughput (compiled BF16 beam): 23.9 users/s at B=1 → 71.5 (B=16) → 76.7 (B=32)
  → 77.9 (B=64); compute-bound above B≈8 (≈63% of A6000 BF16 peak).
- 01:30 — Added: trie-level prefill fusion (prompt + all first-level trie nodes in one forward;
  rank-identical, FP32 score diff ≤3e-5), FP8 W8A8 (L40S), roofline microbenchmark,
  Industrial & Scientific checkpoint (48 / 2,295 / 3,670 trie nodes; 4,533 test users),
  exhaustive FP32 dumps for both datasets (queued), λ-rescoring with cross-fitting.
- 01:40 — sweepA: new engine FP32 (graph, uncompiled) **98.8 ms** vs prior FP32 stack 114.8;
  forced-suffix fusion is worth **10 ms** (BF16: 51.3 separate → 41.3 fused); dropping the
  suffix (3 forwards) 30.3 ms.
- 01:40 — **Quality vs the FP32 MiniOneRec reference decode (paired bootstrap, 4,866 users):**
  BF16 beam ΔHR@10 −0.10 pt [−0.27, +0.06] (free); no-suffix beam ΔNDCG@10 +0.36 [+0.13, +0.61];
  prefix-filtered exact N=16 ΔHR@10 **+0.70 [+0.12, +1.29]**, ΔHR@5 +0.66 [+0.29, +1.05],
  ΔNDCG@10 +0.36 [+0.13, +0.60]; N=32 ΔHR@10 +0.64 [+0.21, +1.09]. Under no-suffix scoring
  beam-50 and certified exact search give identical HR@10 (0.16235) → there, the scoring rule,
  not the search, is what matters. Table: `results_table.md`.
- 01:45 — Bug caught locally before GPU time: Industrial has 48 first codes < beam 50
  (HF semantics: extra beams dead). Fixed.
- 01:50 — **Full MiniOneRec semantics, BF16: 27.9 ms median / 28.0 p95** (fused first level +
  Triton shared-prefix attention + compile), vs 56.5 ms prior BF16 stack, 869 ms HF. Full-test
  HR@10 0.15886 = FP32 reference exactly; NDCG@10 0.1249 vs 0.1247. Triton attention is ~2×
  closer to FP32 than the PyTorch BF16 attention path (self-check max err 0.06 vs 0.16).
- 02:05 — **Certified exact decoding, no forced suffix, BF16: 19.8 ms / 19.9 p95** (2 forwards:
  prompt+first-level tree, then 64 prefix rows). Top-10 certified globally optimal for 94%,
  top-20 for 86%. HR@1 0.0984 / HR@10 0.1624 / NDCG@10 0.1283.
- 02:05 — **HEADLINE: trie-renormalized scoring + certified exact decoding, BF16: 16.1 ms /
  16.25 p95.** Top-50 certified globally optimal for 99.9% of users (top-1..20: 100%). Quality
  above the FP32 MiniOneRec reference on every metric: HR@1 0.0984 (0.0941), HR@5 0.1408 (0.1404),
  HR@10 0.1646 (0.1589), NDCG@10 0.1289 (0.1247), HR@50 0.2390 (0.2361). 54× faster than HF
  generate (869 ms), 3.5× faster than the prior BF16 stack (56.5 ms), 7.1× faster than the
  prior FP32 exact stack (114.8 ms). Restricted (600-row SID) LM head: no 152k-row head at all.
- 02:05 — GemLite INT8 weight-only (uncompiled, torch attention): 37.1 ms vs BF16 41.3 — only
  10%; INT8 kernels at M≈50 are not bandwidth-efficient. Compiled variants pending.
- 02:20 — Paired bootstrap vs FP32 reference (4,866 users): fused+Triton beam (27.9 ms) ΔHR@10
  +0.00 [−0.10,+0.10] (equivalent); certified exact no-suffix (19.8 ms) ΔHR@1 +0.43 [+0.14,+0.72],
  ΔNDCG@10 +0.35 [+0.12,+0.59]; **certified exact trie (16.1 ms) ΔHR@1 +0.43 [+0.12,+0.76],
  ΔNDCG@10 +0.42 [+0.10,+0.74], ΔHR@10 +0.58 [−0.02,+1.15]**; GemLite INT8 ΔHR@10 −0.10
  [−0.29,+0.08] (equivalent).
