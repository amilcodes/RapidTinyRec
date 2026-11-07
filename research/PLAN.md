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

## Status log (chronological, night of 2026-11-06/07)

- Speedrun engine (eager) reproduces FP32 reference 50/50 ranks on MPS
  (scores within 4e-5), batched + padded too. FP16 needs q pre-scaling (Qwen bias overflow);
  then FP16 is ~20× closer to FP32 than BF16 (max score error 0.05 vs 0.9).
- A6000 smoke, BF16, CUDA graph, no compile: **41.3 ms median / 41.9 p95**
  (prior BF16 stack 56.5). Full Office test (4,866): HR@10 0.158, NDCG@10 0.124.
  Throughput 23.9 users/s (B=1), 57.7 (B=8).
- Kernel profile (BF16, eager, per request): ~24 ms GEMM (≈65% of A6000 peak BW),
  ~14 ms in ~4,000 tiny elementwise kernels (RMSNorm/RoPE/attention glue), ~3–4 ms full-vocab
  LM head (152k rows, 3 calls). ⇒ levers: fusion (compile + Triton beam attention), weight
  bytes (INT8/INT4), restricted LM head.
- Exact search (BF16, single-bound variant, N=32 items): **32.4 ms** vs beam 41.3 ms
  (3 forwards vs 4); full test HR@10 **0.165 vs 0.158**, NDCG@10 0.127 vs 0.124. Top-1
  certified for 96% of users, top-10 for 5% (U2 bound is loose under full-vocab scoring).
  BF16 noise (max score error vs FP32 ≈3.6) makes BF16 certification/exhaustive comparisons
  unreliable → science runs moved to FP32.
- Local FP32: under trie-renormalized scoring the 3-stage exact search certifies
  top-20 for every test user; under full-vocab scoring certification mostly stops at
  top-1/5 because item scores fall off steeply (e.g. −0.04, −11.8, −24.4) and beam missed a
  better 3rd item (−24.4 vs beam −27.2) — real beam search error.
- torch.compile (qkv/post fused) BF16 beam: **35.2 ms** (from 41.3).
- **Search-error-helps signal.** BF16 exact search, full-vocab scoring, paired bootstrap
  over 4,866 users: N=32 HR@10 0.1652 vs N=64 0.1599 (Δ −0.53 pt [−0.95, −0.14]) vs
  N=128 0.1587 (Δ −0.66 [−1.11, −0.23]); NDCG@10 likewise. Items are admitted by the prefix
  bound lp(a)+lp(b|a) and ranked by the full five-term score; admitting more items (closer
  to the true argmax) *hurts*. Hypothesis: the c-level and forced \n/EOS terms of the
  full-vocabulary objective are mis-calibrated for ranking (cf. Stahlberg & Byrne 2019 for NMT).
  Test: FP32 sweep over scoring rules (full / no-suffix / codebook / trie) × (beam / exact).
- **Certificate validated in FP32** (exact N=64 vs brute-force scoring of all 3,444
  SIDs, 100 users): 0 violations among 23 certified-top-10 users; top-1 = global optimum for
  100%. The FP32 beam-50 reference returns a non-optimal top-1 for 3/500 cohort users.
  FP32 full-test quality matches BF16 (HR@10 0.1595 vs 0.1599 at N=64).
- Throughput (compiled BF16 beam): 23.9 users/s at B=1 → 71.5 (B=16) → 76.7 (B=32)
  → 77.9 (B=64); compute-bound above B≈8 (≈63% of A6000 BF16 peak).
- Added: trie-level prefill fusion (prompt + all first-level trie nodes in one forward;
  rank-identical, FP32 score diff ≤3e-5), FP8 W8A8 (L40S), roofline microbenchmark,
  Industrial & Scientific checkpoint (48 / 2,295 / 3,670 trie nodes; 4,533 test users),
  exhaustive FP32 dumps for both datasets (queued), λ-rescoring with cross-fitting.
- sweepA: new engine FP32 (graph, uncompiled) **98.8 ms** vs prior FP32 stack 114.8;
  forced-suffix fusion is worth **10 ms** (BF16: 51.3 separate → 41.3 fused); dropping the
  suffix (3 forwards) 30.3 ms.
- **Quality vs the FP32 MiniOneRec reference decode (paired bootstrap, 4,866 users):**
  BF16 beam ΔHR@10 −0.10 pt [−0.27, +0.06] (free); no-suffix beam ΔNDCG@10 +0.36 [+0.13, +0.61];
  prefix-filtered exact N=16 ΔHR@10 **+0.70 [+0.12, +1.29]**, ΔHR@5 +0.66 [+0.29, +1.05],
  ΔNDCG@10 +0.36 [+0.13, +0.60]; N=32 ΔHR@10 +0.64 [+0.21, +1.09]. Under no-suffix scoring
  beam-50 and certified exact search give identical HR@10 (0.16235) → there, the scoring rule,
  not the search, is what matters. Table: `results_table.md`.
- Bug caught locally before GPU time: Industrial has 48 first codes < beam 50
  (HF semantics: extra beams dead). Fixed.
- **Full MiniOneRec semantics, BF16: 27.9 ms median / 28.0 p95** (fused first level +
  Triton shared-prefix attention + compile), vs 56.5 ms prior BF16 stack, 869 ms HF. Full-test
  HR@10 0.15886 = FP32 reference exactly; NDCG@10 0.1249 vs 0.1247. Triton attention is ~2×
  closer to FP32 than the PyTorch BF16 attention path (self-check max err 0.06 vs 0.16).
- **Certified exact decoding, no forced suffix, BF16: 19.8 ms / 19.9 p95** (2 forwards:
  prompt+first-level tree, then 64 prefix rows). Top-10 certified globally optimal for 94%,
  top-20 for 86%. HR@1 0.0984 / HR@10 0.1624 / NDCG@10 0.1283.
- **HEADLINE: trie-renormalized scoring + certified exact decoding, BF16: 16.1 ms /
  16.25 p95.** Top-50 certified globally optimal for 99.9% of users (top-1..20: 100%). Quality
  above the FP32 MiniOneRec reference on every metric: HR@1 0.0984 (0.0941), HR@5 0.1408 (0.1404),
  HR@10 0.1646 (0.1589), NDCG@10 0.1289 (0.1247), HR@50 0.2390 (0.2361). 54× faster than HF
  generate (869 ms), 3.5× faster than the prior BF16 stack (56.5 ms), 7.1× faster than the
  prior FP32 exact stack (114.8 ms). Restricted (600-row SID) LM head: no 152k-row head at all.
- GemLite INT8 weight-only (uncompiled, torch attention): 37.1 ms vs BF16 41.3 — only
  10%; INT8 kernels at M≈50 are not bandwidth-efficient. Compiled variants pending.
- Paired bootstrap vs FP32 reference (4,866 users): fused+Triton beam (27.9 ms) ΔHR@10
  +0.00 [−0.10,+0.10] (equivalent); certified exact no-suffix (19.8 ms) ΔHR@1 +0.43 [+0.14,+0.72],
  ΔNDCG@10 +0.35 [+0.12,+0.59]; **certified exact trie (16.1 ms) ΔHR@1 +0.43 [+0.12,+0.76],
  ΔNDCG@10 +0.42 [+0.10,+0.74], ΔHR@10 +0.58 [−0.02,+1.15]**; GemLite INT8 ΔHR@10 −0.10
  [−0.29,+0.08] (equivalent).
- INT4 weight-only (GemLite A16W4, RTN g128): no speedup (beam 39.4 ms uncompiled vs 41.3 BF16;
  certified trie 16.5 / p95 19.6 ms vs 16.1 / 16.3 BF16) and lower fidelity (beam top-1 = FP32 for
  373/500 vs 471 BF16; certified trie NDCG@10 0.1274 vs 0.1289).
  INT8 (GemLite A16W8): quality-neutral, only 3–8% faster compiled. Negative result: at 50–200
  rows per forward these weight-only kernels do not beat cuBLAS BF16 on the A6000.
- Kernel profile of the compiled engines (eager replay of the same kernels): GEMMs ≈80% of GPU
  time for both the 27.9 ms beam and the certified-exact paths (≈75% of peak HBM bandwidth at
  M≈50–216); remaining glue ≈4 ms/request (from ≈14 ms before fusion); Triton beam attention
  ≈0.25–0.33 ms/request; prefill attention (masked SDPA) ≈0.35–0.5 ms. ⇒ at batch 1 the A6000
  BF16 path is within ~2× of the weight-streaming floor; further gains need better skinny-GEMM
  kernels, fewer bytes (FP8 on Ada), or fewer tokens in the first forward (template cache).
- **FP16 dominates BF16 at equal speed** (fused+Triton beam, full semantics): 27.5 vs 27.9 ms;
  vs FP32 reference: top-1 equal 500/500 vs 488; top-10 set equal 484/500 vs 374; rank positions
  equal 91.5% vs 57.7%; identical 50-lists 151 vs 0; max |Δscore| 0.15 vs 1.48. Requires Q
  pre-scaling (Qwen2's QKV biases overflow FP16 attention logits otherwise). Recommendation: serve
  LLM recommenders in FP16, not BF16, when list fidelity matters.
- Scoring rules, beam-50, BF16, full Office test: full-vocab (MiniOneRec) HR@10 0.1589 / NDCG@10
  0.1247; no-suffix 0.1624 / 0.1284; trie 0.1648 / 0.1287; **codebook (FlashRec-style per-level
  softmax) 0.1667 / 0.1299, HR@50 0.2470** — the restricted normalizations production engines use
  for speed are also *more accurate* than the objective the model was trained with.
- Exhaustive FP32 dump, shard 1 of 4 (1,217 users): under full-vocab scoring beam-50 recovers 86%
  of the exact top-10 (top-1 exact 99.8%) with identical HR@10; under codebook/trie scoring beam-50
  is essentially exact (99.6% / 99.9%). So beam's search error is real but not what limits quality.
- **Depth-discounted scoring** (in-sample on that shard, to be validated on MiniOneRec's validation
  split): score = lp(a) + w_b lp(b|a) + w_c lp(c|ab) + w_s (lp(\n)+lp(EOS)); w=(0.75, 0.3, 0.3)
  gives HR@10 0.1668 vs 0.1512 (+10% rel.), NDCG@10 0.1235 vs 0.1159; single λ on the tail
  (w_c=w_s=0.3) cross-fits to +1.15 pt [+0.25, +2.14]. Later SID levels should count less.
  Validation dumps (Office + Industrial, ~1.2k users each) queued for honest tuning.
- torchao INT8 + compile OOMed; skipped.
- **Paired ladder (ladderZ, one allocation, same A6000, 2×500 requests per rung):** FP32 105.0 →
  BF16 one graph 46.3 (separate-suffix variant 55.9) → +compile 37.5 → +fused first level +
  Triton 30.2 → certified exact no-suffix 21.6 → certified exact trie 17.8; beam-50 with trie
  scoring 17.7 (same cost: the certificate is free); exact trie M=32 17.9; prefix-filtered exact
  N=16 full semantics 21.2. This card is ~7–10% slower than earlier allocations of the same GPU
  type (e.g. 30.2 vs 27.9 ms for the same config), so ratios come from the paired ladder and
  absolute bests are reported with that spread.
- Inductor max-autotune (no cudagraphs): 27.7 vs 27.9 ms default — no gain at these shapes.
- FP16 certified exact no-suffix: 19.5 ms, HR@10 0.1619 (BF16 19.8, 0.1624).
- Mechanism (exhaustive FP32, 1,217 users): lp(c|ab) median −0.87/−2.46/−3.05/−3.38/−4.12 for
  1/2/3/4/6 siblings (≈ −log k structural penalty; r = −0.36); median 47% of mass leaks off the
  c-codebook even after a valid prefix; suffix is mixed (target rank up 28% / down 25% of users,
  r = 0.15 with log popularity).
- **FP32 certificate validation (certV) vs brute-force scoring of all SIDs:** trie scoring, 300
  users: 0 violations, top-10 certified 300/300, returned lists = exhaustive optimum at every
  K ≤ 20 (search recall@50 0.99993). No-suffix scoring, 200 users: 0 violations (190 certified at
  top-10), top-1 always optimal. Beam-50 with trie scoring (17.7 ms) gives the same quality as
  certified exact trie (HR@10 0.1650 vs 0.1646): under trie scoring beam is near-exact, and the
  exact decoder adds the proof at no cost.
- **Roofline (A6000, one full forward incl. full LM head + logsumexp, eager kernels in a CUDA
  graph):** BF16 8.1 ms at 1 token, 9.8 at 50, 11.0 at 88, 12.2 at 128, 16.1 at 256, 28.7 at 512,
  53.8 at 1,024 → 50 beam rows cost 1.2× one row; linear beyond ~256. INT8 weight-only (GemLite):
  5.2 ms at 1 token (1.55×), 8.3 at 50 (1.19×), no gain ≥ 88; INT4: 4.4 ms at 1 (1.8×), 8.7 at 50,
  slower ≥ 88. Our forwards carry 50–216 rows, which is why weight-only quantization barely helps.
- **Shared-template K/V cache (69 of 76–121 prompt tokens):** certified exact trie decoding
  **14.36 ms / 15.41 p95** (same quality: HR@10 0.1650, NDCG@10 0.1288, top-50 certified 99.9%);
  full-semantics beam 26.3 ms; exact no-suffix 18.2 ms. **869 → 14.4 ms = 60×.**
  Throughput, certified exact trie + template: 67.8 users/s (B=1), 135.6 (B=4), 145.9 (B=8)
  vs 77.9 users/s saturated for compiled beam-50.
- Exhaustive dumps, half the Office test set (2,433 users), paired bootstrap vs full/beam-50:
  tail weight λ=0.3 (lp(a)+lp(b|a)+0.3·rest) HR@10 0.1709 vs 0.1578 (**+1.31 pt [+0.66, +2.01]**),
  NDCG@10 0.1295 vs 0.1232; 2-fold cross-fit picks λ=0.3 in both folds. Beam-10 > beam-50 under the
  full objective (+0.82 [+0.08, +1.56]); exhaustive vs beam-50 under the full objective −0.12
  [−0.37, +0.12]. Search exactness is not the quality lever; the scoring rule is.
- **Throughput with template cache (A6000, BF16, users/s at B = 1/4/8/16/32):** certified exact
  trie 67.8 / 135.6 / 145.9 / 165.2 / **172.2**; beam-50 full semantics 37.5 / 69.2 / 77.0 / 86.0 /
  91.3. Exact decoding is ~1.9× ahead at every batch size (fewer tokens per user: prompt tail +
  88 trie nodes + 64 prefix rows vs prompt + 88 + 50 + 100 beam rows).
- **Depth-discounted scoring, honest evaluation:** weights tuned on 1,217 MiniOneRec *validation*
  users (grid, max NDCG@10, exhaustive search), evaluated once on 2,434 test users vs the FP32
  reference decode (full, beam-50): full-vocab with w_c=0.5, w_s=0.1 → HR@10 0.1684 vs 0.1578,
  **+1.07 pt [+0.41, +1.73]**, NDCG@10 +0.57 [+0.24, +0.91]; codebook with w_c=0.5 → +0.99
  [+0.21, +1.81] / +0.46 [+0.06, +0.85]; no-suffix full w_c=0.75 → +0.66 [+0.00, +1.27]; trie
  w_c=0.75 → +0.49 (n.s.). w_b=1 chosen everywhere: only the third code (and suffix) is discounted.
  Beam-50 and exact give the same numbers under tuned weights.
- FP16 + template cache: full-semantics beam **26.1 ms with top-1 = FP32 for 500/500** and HR@10
  0.1587 (FP32 0.1589) — the faithful fast path; certified exact trie 14.3 ms (= BF16 14.4);
  beam-50 trie 14.3 ms; exact no-suffix 18.2 ms.
- Whole-layer torch.compile incl. Triton attention: 14.29 vs 14.31 ms — no gain (GEMM-bound).
- **Office, full test (4,866), exhaustive FP32 simulation** (`research/sim_office_full.md`,
  `research/tune_office.md`): under the full objective exact search is slightly *worse* than
  beam-50 (NDCG@10 −0.08 [−0.14, −0.02]); beam-50 recovers 86.4% of the exact top-10. Codebook
  scoring untuned +0.90 [+0.35, +1.48] HR@10; validation-tuned codebook (w_c=0.5) **+1.15 [+0.55,
  +1.73] HR@10, +0.64 [+0.34, +0.96] NDCG@10**, matching the engine (0.1700 / 0.1309).
- **Engine, Office full test: certified exact, codebook, w_c=0.5, template, BF16 = 14.4 ms**; vs
  FP32 reference decode HR@1 +0.35 [+0.04, +0.68], HR@10 +1.11 [+0.53, +1.69], NDCG@10 +0.62
  [+0.32, +0.93], HR@50 +1.50 [+0.86, +2.16]; top-10 certified 99.7%, top-50 92.4%. FP16 same.
- **Industrial (4,533 users):** speed replicates (certified exact no-suffix 13.9 ms vs compiled
  beam 26.5 ms, fused beam 20.0 ms) but the scoring story does not transfer as-is: dropping the
  forced suffix *hurts* (NDCG@10 −0.42 [−0.59, −0.25]); trie ≈ full (+0.13 HR@10, n.s.). The
  scoring rule is dataset-dependent → must be chosen on validation (Industrial tuning pending).
