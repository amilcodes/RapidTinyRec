# Research plan: how fast can LLM-based generative recommendation go?

Working title: **"Beam search is optional: certified exact and roofline-bound inference for
LLM-based generative recommenders."**

## Current best results (2026-11-08)

Batch 1, frozen 500-user Office cohort, FP16, one CUDA graph per request; quality on all test users.

| decoder | A6000 ms | H100 ms | Office HR@10 | notes |
|---|---:|---:|---:|---|
| HF `generate`, FP32 (reference decode) | 868.8 | — | 0.1589 | MiniOneRec beam-50 |
| speedrun engine, FP16, MiniOneRec semantics | 26.2 | 8.5 | 0.1587 | top-1 = FP32 500/500 |
| certified exact (three-stage, M=64) | 14.4 | 5.6 | 0.1650 / 0.1700 | trie / tuned codebook |
| certified exact, trie, multi-child only (M=16) + Triton prompt attention | 13.25 | 4.37 | 0.1652 | no learned parts; FP32 brute-force check: 0 violations |
| + late entry for the third code (bridge, layer 22) | 8.98 | 3.16–3.32 | 0.1669 | bridge trained on validation only |
| **codebook(a,b)+trie(c) scoring, w_c=0.75, + late entry** | **9.02** | **3.08** | **0.1712** | ΔHR@10 +1.23 [+0.66, +1.81] vs reference |
| certified top-500 + late entry | 10.6 | 3.8 | — | MiniOneRec beam-500: 83.2 / 18.1 ms |

Throughput (H100, certified top-10 + late entry): 1,447 users/s at B=64. Late entry is lossless on HR@10 and
NDCG@10 on both datasets (Office 6 of 28 layers, Industrial 10). Details and every number's run: below.

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
- **Industrial (4,533 users):** speed replicates (*measured on an L40S — this job ran on any GPU*:
  certified exact no-suffix 13.9 ms vs compiled beam 26.5 ms, fused beam 20.0 ms) but the scoring story does not transfer as-is: dropping the
  forced suffix *hurts* (NDCG@10 −0.42 [−0.59, −0.25]); trie ≈ full (+0.13 HR@10, n.s.). The
  scoring rule is dataset-dependent → must be chosen on validation (Industrial tuning pending).
- **Industrial certified exact trie: 11.5 ms on L40S** (48 first codes; no template cache) vs compiled
  beam 26.5 ms / fused beam 20.0 ms on the same L40S; top-20 certified 100%, top-50 97.9%; HR@10 0.1582 vs 0.1571 and
  NDCG@10 0.1167 vs 0.1172 for full-semantics beam (equal quality). The Office "narrow prefix
  filter helps" effect does not transfer (N=16 HR@10 0.1458, −1.1 pt). Robust claims: speed with
  equal-or-better quality via certified trie decoding on both datasets; scoring-rule gains beyond
  that are dataset-specific and must be tuned on validation.
- **Industrial, validation-tuned scoring (1,134 valid users → 3,400 test users):** no rule beats
  the original objective significantly (tuned codebook HR@10 +0.35 [−0.26, +0.97], NDCG@10 −0.03);
  untuned codebook/trie ≈ 0; dropping the suffix hurts (−0.56 [−1.03, −0.06] / −0.48 [−0.70, −0.27]).
  ⇒ The Office scoring gains (+1.1 pt HR@10) are dataset-specific. What generalizes: certified
  trie decoding at ~2× fewer forwards with equal quality on both datasets; and the recipe "choose
  the scoring rule on validation" (it picks codebook + w_c=0.5 on Office, ≈ original on Industrial).
- **Logit lens (1,000 users each):** SID codes are decided late. Office: top-1 first code agrees
  with layer 28 for ≤7% of users through layer 19, 45% at 22, 69% at 25; second code (given a)
  55% at 22, 71% at 25. Industrial: 21% at 16, 65% at 22, 77% at 25. Target's first code is in
  the lens top-10 for 44%/61% at layer 22 vs 51%/63% at the output (Office/Industrial).
  Implication: no early exit for the first two codes; depth reduction would need training.
- **Early exit for the c-level forward** (certified trie + template, Office): layers 28/22/18/14/10/6
  → 14.4/13.2/12.2/11.3/10.4/9.4 ms; HR@10 0.165/0.164/0.164/0.164/0.164/0.165 (flat) but HR@1
  0.098/0.091/0.083/0.073/0.066/0.062 and NDCG@10 0.129/0.124/0.121/0.116/0.113/0.111. The third
  code decides the order among siblings, not set membership; early exit is a latency/precision
  trade (−34% latency at −3.6 pt HR@1), not a free win. Certification is w.r.t. the truncated score.
- **Correction / A6000 Industrial numbers (indT):** the indH latencies above were on an L40S
  (job relaxed to any GPU). On the A6000: certified exact trie + template **14.0 ms / 14.4 p95**;
  certified exact codebook + template 14.1 ms (HR@10 0.1593, NDCG@10 0.1168, top-10 cert 94.7%);
  full-semantics beam FP16 + template 25.6 ms (HR@10 0.1584, NDCG@10 0.1179); FP32 graph 96.5 ms.
  ⇒ Industrial A6000: certified decoding 1.8× faster than the fully optimized full-semantics beam
  at equal quality; 11.5 ms vs 14.0 ms is the L40S vs A6000 difference.
- **Industrial, full test (4,533), exhaustive FP32** (`research/sim_industrial_full.md`,
  `research/tune_industrial.md`): beam-50 recovers 93.9% of the exact top-10 under the full
  objective (Office 86.4%); exact vs beam −0.02 pt HR@10 (n.s.). Validation-tuned codebook
  (w_c from the grid) +0.29 [−0.24, +0.84] HR@10, +0.02 NDCG@10 (n.s.); dropping the suffix
  −0.42 [−0.82, −0.02] / −0.46 [−0.65, −0.27]. Confirms: scoring gains are dataset-specific;
  certified decoding is quality-neutral on Industrial and quality-positive on Office.
- **L40S (Ada) results:** full-semantics fused beam BF16 20.1 ms (A6000 27.9); certified exact
  no-suffix 14.0 ms (A6000 19.8). **FP8 W8A8 (dynamic per-token, torch._scaled_mm rowwise) is
  slower**: beam 23.5 ms, exact 15.9 ms; quality ≈ BF16 (HR@10 0.1582 vs 0.1589; top-1 = FP32
  450/500). L40S roofline per forward at M = 1/50/128/256/512/1024: BF16 6.4/8.0/9.0/11.4/16.2/29.4 ms;
  FP8 6.7/8.3/11.6/17.6/28.9/51.9 (activation quantization + kernel overhead dominate); GemLite
  INT8 weight-only 4.3/6.3/7.3/10.5/17.0/30.9 (1.2–1.5× at M ≤ 128 — better than on the A6000).
  Negative result for naive FP8; a fused quant-in-RMSNorm FP8 path is future work.
- **Final paired ladder (ladder2: one allocation, same A6000, 2×500 requests per rung):** FP32
  100.8 → BF16 one graph 42.1 → +compile 35.0 → +fused first level + Triton 28.1 → +FP16 + template
  26.2 (top-1 = FP32 500/500) → **certified exact, trie 14.38 / 15.68 p95** → **certified exact,
  validation-tuned codebook (w_c=0.5) 14.37 / 15.64** (top-10 certified 99.7%; HR@10 0.1700 vs
  0.1589 FP32 reference). Throughput of the tuned decoder: 67.8 (B=1), 160.5 (B=8), 176.2 (B=32)
  users/s. **869 → 14.4 ms = 60.4× (HF FP32 → certified FP16); 56.5 → 14.4 = 3.9× over the prior
  BF16 stack; 114.8 → 14.4 = 8.0× over the prior FP32 exact stack.**

## Status log (2026-11-07/08, second session)

- **Can the second forward go? No, not for free** (`scripts/onepass.py`, exhaustive FP32 Office, 4,866 test
  users, validation-tuned codebook objective). Replacing the per-user third-code term by a user-independent
  per-item table fit on validation: HR@1 0.050 vs 0.098 (−4.8 pt [−5.5, −4.1]), NDCG@10 −2.9 pt. Item identity
  explains 77% of the variance of s(c|ab) overall but only 42% within each user's top-50; sibling order agrees
  72%. The third code carries user-specific ranking signal for the top slot.
- **Speculating the second forward's prefixes before the first forward does not work** (`scripts/speculate.py`,
  356 Office users present in both validation and test). To certify top-10 a predictor must cover every prefix
  with U2 > s_10. Best hit rate at 128 guessed prefixes: 29% (previous request's top prefixes + history
  prefixes); history alone 6%, popularity 4%. The previous request's top-10 overlaps the next one by 31%.
- **Certificate expansion sets are ~K under trie scoring** (`research/kscale.json`, both datasets). Prefixes
  with U2 > s_K, median / p90 / p99: Office trie K=10 10/12/14, K=100 104/111/117, K=500 489/510/524,
  K=1000 932/960/979; Industrial similar. Codebook (w_c=0.5) has a heavier tail (K=10 p99 35, K=500 p99 1,058).
  Under the full-vocabulary objective the bound is loose: K=10 p90 1,715 of 2,488 prefixes.
- **Single-child prefixes need no second forward under trie normalization** (s(c|ab)=0 exactly). Office: 581
  of 2,488 prefixes have ≥2 children (1,537 of 3,444 items). Multi-child prefixes needed, median/p90/p99:
  K=10 5/8/10, K=50 22/31/38, K=100 43/55/64, K=500 175/198/213 (Industrial K=500 205/238/259). One-forward
  certification is still rare (K=1: 44%, K=10: 0.5%). New engine `ExactTrieEngine` (`--engine exacttrie`)
  expands only multi-child prefixes; matches ExactEngine3 rank-for-rank and score-for-score on MPS FP32.
- **Large-K search error (exhaustive FP32 simulation, `research/beam_recall_largeK.json`).** Recall of the exact
  top-W by beam-W: full objective Office 44% (W=10), 49% (50), 53% (100), 64% (500); Industrial 62/67/69/79%.
  Trie objective: Office 84/95/96/99.6%; Industrial 87/95/97/99.9%.
- **The beam-search curse grows with list length** (`research/largeK_hr.json`, paired bootstrap vs the
  MiniOneRec decode = full objective, beam-W). Exact search under the full objective: Office HR@50 −1.8 pt
  [−2.5, −1.1], HR@100 −2.5, HR@200 −3.5, HR@500 −4.3 [−5.3, −3.2]; Industrial −0.1 to −0.7 (n.s.). Codebook
  w_c=0.5, exact: Office HR@100 +2.3 [+1.5, +3.0], HR@500 +2.4 [+1.4, +3.3]; Industrial HR@200 +0.9 [+0.1, +1.6],
  HR@500 +0.9 [+0.0, +1.8]. Trie: Office HR@500 +1.0, Industrial −1.6. Beam search's prefix-first pruning acts as
  an implicit depth discount; with an explicit discount, exactness stops hurting.
- **Validation-tuned trie family (`research/hybrid_rule.json`).** w_c chosen on validation = 0.75 for trie and
  for a codebook(a,b)+trie(c) hybrid. Office HR@10 vs reference: codebook +1.15, hybrid +0.92 [+0.33, +1.52],
  trie +0.76 [+0.16, +1.34]; Industrial all +0.29 (n.s.). The trie rules need 9/35/204 second-forward rows (p99)
  for K=10/50/500 vs 35/180/1,058 for codebook.
- **H100 (gpu-debug, one allocation, 2×500 requests per rung, Office):** FP32 graph 48.3 → BF16 graph 20.1 →
  +compile 12.9 → +fused first level + Triton 8.50 → +FP16 + template 8.48 (top-1 = FP32 500/500) → **certified
  exact trie 5.57 / 5.74 p95; validation-tuned codebook 5.66 / 5.76** (HR@10 0.1697). FP8 W8A8 6.01 ms (slower
  again, quality equal). Throughput, tuned codebook: 178 / 726 / 904 / 920 users/s at B = 1/8/32/64.
  Profile: 2,555 kernels and 5.9 ms GPU time per request; GEMMs ≈3.1 ms (≈50% of H100 HBM bandwidth), glue
  ≈2.8 ms (≈200 copy kernels for K/V cache writes, template concat, repeat_interleave, masked SDPA). On H100
  the engine is kernel-count-bound; the weight-streaming floor for two forwards is ≈1.6 ms.
- **Skinny GEMMs on the A6000 (`scripts/gemm_bench.py`):** per forward (28 layers, GEMMs only) cuBLAS 4.84 /
  5.78 / 6.58 / 7.19 ms at M = 16 / 64 / 140 / 216 rows; a tuned split-K Triton kernel only helps at M ≤ 64
  (5.28 ms at M=64, 9%). At M=140 the gate/up GEMM already runs at ≈57 TFLOPS: the A6000's first forward
  (prompt tail + 88 trie nodes) is compute-bound, so fewer rows, not better kernels, is the lever there.
- **Late entry for the third-code forward (`scripts/late_entry.py`, Office, 2,000 test users, codebook
  w_c=0.5, Quadro RTX 6000 FP16).** Prefix rows enter at layer e with an input built from the parent node's
  residual stream at layer e. Same-cost comparison at 6 remaining layers (e=22): full forward HR@1 0.140 /
  NDCG@10 0.175; ridge bridge (h_a, E[b], E[a] → h_b, fit on 600 held-out prompts, no labels) 0.132 / 0.169;
  early exit (first 6 layers) 0.086 / 0.145; parent state only 0.113 / 0.152 (adding E[b]−E[a] changes
  nothing). The bridge recovers most of what early exit loses but leaves −0.8 pt HR@1; its relative residual is
  ≈0.62 at every depth. A trade-off (F2 at 6/28 layers), not a free win.
- **Distilled late-entry bridge (`scripts/train_bridge.py`, `rapidtinyrec/speedrun/bridge.py`).** Bridge =
  ridge(h_a, E[b], E[a]) + MLP([rms(h_a), E[b], E[a]]) (2,048 hidden, output layer zero-initialized), trained
  for 3 epochs (≈30 s on an H100) on 3,000 Office *validation* prompts by KL between the full-depth and the
  late-entry third-code distributions (no item labels). Held-out KL 2.33 (ridge) → 0.088. Entry layer 22 (the
  second forward runs 6 of 28 layers), all 4,866 Office test users, codebook w_c=0.5, FP16, vs full depth:
  HR@1 0.0968 vs 0.0980 (−0.12 pt [−0.37, +0.12]), HR@10 0.1693 vs 0.1706 (−0.12 [−0.35, +0.08]), NDCG@10
  0.1304 vs 0.1312 (−0.08 [−0.22, +0.04]); top-1 identical for 91.7% of users, top-10 overlap 92.2%. Ridge only:
  −0.51 / −0.58 / −0.54 pt. Early exit at the same cost: −3.68 / −0.84 / −2.11 pt. Still above the MiniOneRec
  reference decode (HR@1 0.0941, HR@10 0.1589, NDCG@10 0.1247). **The third code does not need the lower 22
  layers.**
- **Large K, latency (batch 1, FP16, template, Triton, compiled; cohort of 500).** `ExactTrieEngine`
  (multi-child prefixes only): A6000 top-10 (M=16) 13.57 ms, top-50 (M=48) 14.08, top-100 (M=80) 14.70,
  top-200 (M=128) 15.06, top-500 (M=256) 17.95; certification 100% at every K on the cohort. H100: 5.46 /
  5.30 / 5.66 / 5.75 / 6.06 ms. Beam with trie scoring and the fused first level is the same two-forward
  structure without a proof (select W prefixes by U2, one c-forward): H100 beam-50/100/200/500 5.55 / 5.75 /
  6.26 / 7.25 ms.
- **Large K vs MiniOneRec semantics (full-vocabulary scoring, forced suffix, beam-W).** H100 batch 1:
  beam-100 9.90 ms, beam-500 18.12 ms vs certified top-100 5.66 / top-500 6.06 ms (3.0× at K=500, with a
  proof and +2.4 pt HR@500 for the tuned codebook rule on Office). A6000: beam-trie W=50/100/200 14.24 /
  15.10 / 17.00 ms vs certified top-50/100/200 14.08 / 14.70 / 15.06 ms.
  **Throughput at K=500 (H100, users/s):** certified top-500 173 / 374 / 414 / 450 (B = 1/4/8/16);
  beam-500 trie 132 / 229 / 253 (B = 1/4/8); MiniOneRec beam-500 54 / 68 / 71 → **6.4×**. K=100: certified
  185 / 689 / 858 (B = 1/8/32). Top-10 trie engine (M=16): 182 / 837 / 1,223 users/s (B = 1/8/32), vs 904 for
  the codebook three-stage engine at B=32.
- **Bridge depth frontier (Office, all test users, vs full-depth third code):** entry 22 (6 layers) ΔHR@1
  −0.12 [−0.37, +0.12], ΔNDCG@10 −0.08 (n.s.); entry 24 (4 layers) −0.27 [−0.58, +0.02] / −0.09 (n.s.);
  entry 26 (2 layers) −0.45 [−0.78, −0.10] / −0.24 [−0.43, −0.06]. Ridge-only and early exit are worse at
  every depth (early exit −3.3 to −3.7 pt HR@1).
- **Late entry in the engine (H100, one allocation, `runs/lateLH`):** certified exact, codebook w_c=0.5, M=64:
  5.57 ms → late entry at 22: **4.17 ms** (full-test HR@1 0.0968 / HR@10 0.1693 — identical to the offline
  bridge evaluation) → + whole-layer compile **3.84 ms**. Trie engine M=16: 5.03 ms without late entry.
- **Entry 18 (10 layers for the third code) is lossless:** ΔHR@1 +0.06 [−0.12, +0.27], ΔNDCG@10 −0.00
  [−0.12, +0.12].
- **Late entry for the first-level nodes is a trade-off** (`scripts/train_bridge_b.py`): the 88 a-node rows
  join forward 1 at layer 22 from bridge_b(h_22(last prompt token), E[a], E[last token]); s(a) stays exact.
  Held-out KL 2.16 → 0.139. Alone (exact third code): ΔHR@1 −0.21 [−0.55, +0.12], ΔHR@10 −0.27 [−0.74, +0.18],
  ΔNDCG@10 −0.25 [−0.51, −0.01]; only 61% of top-10 prefixes agree with full depth. Stacked with the c-bridge
  (trained on full-depth parent states): ΔHR@1 −1.79. The second code decides set membership and needs its
  lower layers more than the third code does.
- **Triton prompt attention for the fused first forward (`--tree-fast`, H100, `runs/fastH`):** trie engine
  M=16: 5.47 → **4.56 ms**, identical full-test quality (HR@10 0.1652); + late entry 22: **3.56 ms**
  (HR@1 0.0962, HR@10 0.1679); + whole-layer compile: **3.31 ms / 3.40 p95**. Late entry without the fast
  first forward: 4.16 ms (3.79 with compile-layer). Self-check of the prompt kernel vs FP32: max err ≤ 0.009.
- **A6000 MiniOneRec beam baselines:** full-semantics beam-100 30.6 ms, beam-500 83.2 ms (vs certified top-100
  14.7, top-500 17.95 ms: 2.1× / 4.6×); trie beam-500 25.95 ms.
- **Industrial bridges (validation-trained, all 4,533 test users, vs full depth HR@1 0.0847 / HR@10 0.1613 /
  NDCG@10 0.1181):** entry 22: ΔHR@1 −0.31 [−0.60, −0.02], ΔHR@10 −0.22 [−0.51, +0.07], ΔNDCG@10 −0.29
  [−0.44, −0.13]; entry 18: −0.35 [−0.62, −0.11] / −0.04 / −0.18 [−0.33, −0.04]; ridge −0.8 to −1.0; early
  exit −2.4. Office entry 14: −0.02 / +0.14 / +0.05 (lossless). Late entry is lossless on Office from 10
  layers and costs ≈0.2–0.35 pt on Industrial; retraining with prefix-weighted KL queued (`runs/bridgeW`).
- **FP32 brute-force validation of `ExactTrieEngine` (M=16, 300 cohort users, `runs/certT`):** 0 certificate
  violations; top-10 certified 300/300; top-1/5/10 equal to exhaustive scoring of all 3,444 SIDs for every
  user (search recall@20 0.9985, @50 0.919 — M=16 is sized for top-10/20).
- **One-pass trie engine (`OnePassTrieEngine`, every item scored in one forward, H100):** top-50 4.11 ms,
  top-500 4.37 ms; full-test quality identical to the two-pass certified decoder (HR@10 0.16790 both — the
  certified two-pass top-50 equals the exhaustive optimum). Two-pass with late entry is faster: top-10 3.37 ms,
  top-500 **3.81 ms** (H100, fast first forward + compile-layer, late entry 22).
- **A6000 late entry (`runs/lateLA`, paired):** codebook three-stage 14.33 → **9.68 ms** (HR@1 0.0968, HR@10
  0.1695); trie M=16 13.70 → **9.46 ms** (HR@10 0.1681). Triton prompt attention on the A6000: 13.65 → 13.26 ms
  (GEMM-bound GPU; small gain). Throughput with late entry: 101 / 235 / 256 users/s (B = 1/8/32).
- **H100 throughput, fast first forward + late entry + compile-layer:** 301 / 1,120 / 1,412 / **1,447** users/s
  (B = 1/8/32/64).
- **Better bridge training fixes Industrial:** prefix-weighted KL (half uniform over the 64 rows, half the
  softmax of their U2 bounds), 4,300 validation prompts, 6 epochs. Industrial entry 18: ΔHR@1 −0.07 [−0.22,
  +0.09], ΔHR@10 −0.04 [−0.24, +0.18], ΔNDCG@10 −0.04 [−0.13, +0.05] (was −0.35 / −0.04 / −0.18); top-1 = full
  for 91.6%. Late entry is lossless on both datasets with 10 layers.
- **Office bridge with prefix-weighted KL (4,600 validation prompts, 6 epochs), entry 22:** ΔHR@1 +0.27
  [+0.12, +0.45], ΔHR@10 −0.04 [−0.25, +0.16], ΔNDCG@10 +0.10 [−0.01, +0.20]; top-1 = full 94.4%; HR@10 0.1702.
- **H100 final kernel work (`runs/storeH`, paired):** FP16 bridge 3.24 ms (from 3.31 with the FP32 bridge);
  compiled K/V stores (`--fuse-store`) **3.16 ms / 3.23 p95**, identical quality (HR@10 0.16790); profile 1,475
  kernels, 3.47 ms GPU time per request.
- **A6000 (`runs/fastA`):** trie M=16 13.65 → Triton prompt attention 13.26 → + late entry 22 **9.06 ms**
  (HR@10 0.1677) → + compile-layer 9.07; late entry 18: 9.87 ms (HR@10 0.1685); codebook three-stage + fast +
  late 22: 9.31 ms (HR@10 0.1691).
- Graph-capture failures seen twice on the H100 with the FP32 bridge at batch > 1 (one-pass throughput at B=8;
  entry-18 quality pass) while other jobs shared the node and the inductor cache reported stale NFS handles;
  not reproduced with the FP16 bridge so far.
- **A6000 certified top-500 + fast first forward + late entry: 10.58 ms** (vs 17.95 ms without late entry; vs
  83.2 ms for MiniOneRec's beam-500 → 7.9×). A6000 throughput (trie M=16 + fast + late 22): 105 / 239 / 258
  users/s at B = 1/8/32.
- **Final paired runs, Office (`runs/finA` A6000, `runs/finH` H100; FP16, Triton prompt attention; H100 also
  compile-layer + fused stores; late entry = prefix-weighted bridge at layer 22; all 4,866 test users; Δ vs the
  FP32 MiniOneRec reference decode, paired bootstrap):**
  | decoder | A6000 ms | H100 ms | HR@1 | HR@10 | NDCG@10 | ΔHR@10 vs reference |
  |---|---:|---:|---:|---:|---:|---|
  | trie M=16, no late entry | 13.25 | 4.37 | — | 0.1652 | — | |
  | trie + late entry | 8.98 | 3.32 | 0.1007 | 0.1669 | 0.1307 | +0.80 [+0.23, +1.40] |
  | trie w_c=0.75 + late entry | 8.99 | 3.08 | 0.1005 | 0.1697 | 0.1320 | +1.07 [+0.47, +1.64] |
  | **codebook(a,b)+trie(c), w_c=0.75 + late entry** | **9.02** | **3.08** | 0.1009 | **0.1712** | **0.1327** | **+1.23 [+0.66, +1.81]** |
  | codebook w_c=0.5, three-stage M=64 + late entry | 9.17 | 3.10 | 0.1007 | 0.1702 | 0.1322 | +1.13 [+0.53, +1.71] |
  Headline: ΔHR@1 +0.68 [+0.33, +1.05], ΔNDCG@10 +0.79 [+0.48, +1.11], ΔHR@50 +1.05 [+0.39, +1.71]; top-10
  certified 100%. H100 latencies of configurations with the same work differ by up to ≈7% within one job.
