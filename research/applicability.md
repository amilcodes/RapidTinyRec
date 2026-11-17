# Where certified, late-entry decoding applies in deployed generative recommenders

How the method in `research/paper/main.tex` maps onto semantic-ID generative recommenders that are deployed or
published by industry. The facts about other systems come from their primary sources (arXiv HTML, official
repositories and blogs, read 2026-11-17); "n/s" means the source does not say. The facts about our method come from
the measured runs in `research/PLAN.md` (sessions 5–7).

## 1. What the method needs, and what we measured

| Requirement | Why | Measured evidence (Office, H100, FP16, batch 1) |
|---|---|---|
| **R1. Autoregressive SID decoding with per-level log-probabilities** | Bounds rely on every scored term being ≤ 0 | All results. Raw-logit scores (NVIDIA's default) do not qualify, but its masked-logprob mode does |
| **R2. A known set of valid items (trie) at serving time** | The certificate is relative to that set | Real catalog and synthetic catalogs up to 1M SIDs |
| **R3. Few levels (3 as built)** | Two forwards cover 3 levels. More levels need one budgeted pass per level, which is built in principle (`HierTrieEngine` budgets level 1) but not for L > 3 | 3 levels only |
| **R4. A first level that is small, or a budget on it** | Forward 1 has one row per first code | +1.6 µs per first code with bridges (2.64 → 3.47 ms, 88 → 600 codes). With a budget of 32: flat 3.2 ms, but only 61% of top-10s certify on the real 88-code distribution |
| **R5. Restricted (codebook/trie) normalization for a flat cost** | The bound is tight only then | Trie/cbtrie: 100% certified at M = 16. Full vocabulary, no suffix: 94% at M = 64; median 5.2 ms, p99 17–28 ms with exhaustive fallback |
| **R6. No forced suffix scored** | The suffix is not bounded by the prefix | With MiniOneRec's newline+EOS suffix: 89% certified at 16.3 ms (≈ 2× beam) |
| **R7 (late entry only). A deep decoder** | Bridges skip the lower layers | 28-layer 1.5B, 36-layer 3B, 28-layer 7B. Bridges are distilled per model in 1–15 min, without labels |
| **Benefits from long candidate lists** | Expansion ≈ 0.4–1.1·K prefixes, not beam width × levels | Certified top-500: 2.92 ms vs 7.2 ms (trie beam-500), 18.1 ms (MiniOneRec beam-500). 1M SIDs, top-50: 3.19 ms at M = 512 |
| **Benefits from eligibility filters and additive item terms** | Exact eligible / fused top-K under the same certificate | 5% eligibility: filtering the LLM's top-50 recovers 30.5% of the exact eligible top-10. SASRec term +1.36 HR@10 |
| **Long contexts** | The prompt pass is shared with beam | Cached history 1k/4k/8k: 2.0× beam. Uncached 3k: 1.5–1.7× latency, 1.6× throughput (prefill-bound) |

## 2. Deployed and industrial systems

| System | SID: levels × codebook | Catalog | Beam / K | Valid-item constraint | Score normalization | Model / decoder depth | Context | Value or eligibility terms |
|---|---|---|---|---|---|---|---|---|
| **OneRec** (Kuaishou, short video, end to end) | 3 × 8,192 (RQ-Kmeans) | billion-scale | V1 beam 128; K = 512 deployed | **None**: a training-time format reward gets legality to 95% | n/s | Enc-dec MoE, 1B deployed (24 layers at 2.6B) | 20 + 256 items + lifelong compressed to 2,000 | Reward-based selection after generation |
| **OneRec-V2** | 3 SIDs, codebook n/s | n/s | beam 512 | n/s | n/s | Lazy decoder-only, 1B (18 layers in that config) | 3,000 tokens | User-feedback RL only |
| **OpenOneRec** (open release) | 3 × 8,192 | 15.3M items, not 1:1 | eval beam 32 | **None** (vLLM beam search, mapped to items afterwards) | full vocabulary | Qwen3 1.7B / 8B | 30–458 items on average | — |
| **GR4AD** (Kuaishou ads) | 16,384 / 4,096 / 1,024 (last level hashed) | n/s | beams 512³, dynamic 128-256-512; K = last beam | n/s | softmax per level vocabulary (≈ codebook) | 0.16B, 9 decoder layers (6 shared via LazyAR) | n/s | **eCPM token decoded after the SID, used to re-rank; eCPM RL reward** |
| **GRACE** (Meta ads retrieval) | 4 × 512 | 30M SIDs (clusters of ads) | beam 1,024, dynamic (1, 512, 1,024, 1,024) | **Trie + targeting mask at decode time** | log-probs masked to −∞, renormalization n/s | Enc-dec, **3-layer decoder** | 128 tokens | **Targeting (geo, age, …) as decode-time constraints**; no bid term |
| **STATIC** (YouTube Home, candidate source) | **8 × 2,048** | 100M–1B corpus, ~20M constrained | beam 70 | Trie as CSR matrix, every step | **full-vocabulary log-softmax, then mask** | Gemini-based 3B dense | n/s | **Freshness constraint (last 7 days) inside decoding** |
| **xGR / xLLM** (JD e-commerce) | 3 tokens, codebook n/s | n/s | beams 128–512, K = beam | Valid-item mask, every step | **mask before softmax (≈ trie normalization)** | Qwen3 0.6–4B, OneRec 0.1–3B | 1k–3k tokens | none |
| **UniSGR / STARK** (Lazada feed) | 3 × 8,192 | n/s (24% collisions) | beams (512, 512, 1,024) | n/s | n/s | Enc-dec, sparse MoE decoder 0.2–2B | n/s | Downstream ranking head; business weights in training |
| **NEZHA** (Taobao search ads, recall) | 3 × 512 (public; production differs only in codebook size) | valid IDs ≈ 0.1% of code space | beam 512; K up to 1,000 | **Final step only** (hash set) | cumulative probability, scope n/s | Llama-1B / Qwen3-0.8B scale | query + history | Revenue +1.2%; no bid term described |
| **GLIDE** (Spotify podcasts) | 4 × 256 | n/s, not 1:1 | beam 30 → 30 candidates | **None** (lookup table) | n/s | Llama 3.2 1B | SIDs + 1 soft user token | Downstream ranker; playability/locale tie-breaker |
| **Netflix** (AR vs MTP study) | 4 × 1,024 | n/s | n/s | Trie as logit mask every step; per-request eligible catalog | length-normalized log-prob (fixed length) | Qwen3-8B / 0.6B | long member history, prefix KV shared across beams | **Per-request eligibility enforced in decoding** |
| **NVIDIA SID-GR example** | 3 decode steps, codebook n/s | n/s | beam 256 | Tries / masks (untested on real large catalogs) | **default raw logits**; optional mask → log-softmax | Qwen3-1.7B | 1k / 5k tokens, no prefix reuse | — |
| TIGER / LC-Rec / MiniOneRec (academic) | 3 × 256 (+1) / 4 × 256 / 3 × 256 | 12k–21k items | beam 20–50 | after / at decode | full or n/s | T5 4+4 layers / LLaMA-7B / Qwen2-1.5B | 20 items | — |
| PinRec (Pinterest) | **no SIDs** (dense embedding generation + kNN) | — | — | — | — | 12-layer 100M | past year | outcome conditioning |

What these systems share:
- **No system scores a suffix or EOS.** Every one decodes a fixed number of SID levels. Our most expensive case (R6) does not occur in deployment.
- **Normalization is restricted wherever it is stated, with one exception.** GR4AD normalizes per level vocabulary, xGR masks before the softmax, and GRACE has its own 512-token SID vocabulary. STATIC is the exception: full-vocabulary log-softmax, then mask.
- **First levels are wide: 512 to 16,384 codes.** Only the academic setups have 256 or fewer. This is the main mismatch with how our engine is built (R4).
- **Candidate lists are long: K = 30 to 1,000, with beams of 70 to 1,024.** Our cost scales with K rather than with beam width × levels (a strength).
- **Eligibility is enforced inside decoding** in GRACE (targeting), STATIC (freshness) and Netflix (per-request catalog). Exact constrained top-K is a direct fit.
- **Value terms are applied outside the decoder.** GR4AD uses an eCPM token to re-rank, OneRec a reward-based selection step, UniSGR a downstream ranking head. That is the beam-then-rerank pattern whose limits we measured.
- **No source describes caching user-history KV across requests.** Contexts run 1k–5k tokens (NVIDIA, xGR, OneRec-V2), and NVIDIA explicitly runs without prefix reuse. Uncached is the realistic default; our 2× cached-history result applies only where a KV cache exists. KV caching for long histories does exist in ranking (Meituan MTServe, Huawei RelayGR).
- **Serving is throughput under a P99 budget**: 100–200 ms at hundreds of QPS per L20/H800/GH200. Batch-1 latency matters less than the throughput gain and the tail.

## 3. Verdict by system class

**Directly applicable (as built or with configuration).** These are LLM-based SID retrievers with 3 levels, a first level of up to a few hundred codes, valid-item constraints and restricted normalization.
- **Who.** The academic setups (TIGER-style, LC-Rec, MiniOneRec, the 3B/7B MiniOneRec backbones we ran) and NEZHA's public setting (3 × 512).
- **What carries over:**
  - certified exact top-K at lower cost than beam;
  - late entry on deep LLM decoders;
  - exact eligibility and fusion;
  - certified lists of up to hundreds of items.
- **xGR also qualifies on the decoding semantics.** It uses 3 levels and mask-before-softmax, which is the trie normalization our bounds need. Its codebook size is not stated, so whether its first level is narrow is unknown.

**Applicable with an extension we built but which certifies less.** These systems have wide first levels: OneRec, OpenOneRec and UniSGR (8,192), GR4AD (16,384), NEZHA in production (≥ 512).
- **Two-forward engine:** forward 1 would carry 512–16k rows. Extrapolating our measured slope, that is about +1–26 ms.
- **First-level budget (`HierTrieEngine`):** it caps that cost, measured flat at 3.2 ms up to 600 codes. But the certificate then depends on how concentrated the first-level prior is, and on our model only 61% of top-10s certify with 32 of 88 first codes.
- **The asymmetry to remember:** beam search scores every first code in one LM-head row at step 1, so wide first levels cost beam nothing at that step.
- **What it means here:** the method buys exact top-1/top-5 and late-entry speed. Full certification of a 500-item list would need a first-level budget we cannot estimate without such a model.
- **OneRec and OpenOneRec need a valid-item trie added first.** They decode unconstrained and fix legality through training, though STATIC and FlashTrie show tries are cheap at 20M–800M items.

**Applicable for exactness, not for speed.**
- **Deep SIDs:** STATIC has 8 levels, GRACE, GLIDE and Netflix have 4. Hierarchical certified search needs one pass per level, the same number of sequential passes as beam. The gains would come only from per-level budgets below the beam width (STATIC: beam 70) and from exactness.
- **Shallow decoders: GRACE (3 layers) and GR4AD (9).** They leave little for late entry; GR4AD's LazyAR is already a trained version of the same idea.
- **STATIC's full-vocabulary-then-mask scoring** is our "full, no suffix" case: beam-level median cost with a heavier tail.

**Where the parts fit regardless of speed:**
- **Eligibility.** GRACE's targeting, STATIC's freshness and Netflix's per-request catalog all enforce constraints inside decoding. Certified search returns the exact eligible top-K. Post-filtering a beam list recovered only 30.5% of it at 5% eligibility, and GRACE itself reports a 23.6% → 40.4% target-match pass rate from moving targeting into decoding.
- **Value terms.** GR4AD's eCPM re-ranking, OneRec's reward-based selection and UniSGR's ranking head all score after the beam has pruned. An additive value term (for example log eCPM) inside the certified search changes which items are retrieved; that is exactly what reranking cannot do (+2.26 HR@50 in our runs).

**Not applicable.**
- **Dense-embedding generators** (PinRec): there are no SIDs to bound.
- **Parallel, MTP or diffusion SID decoders** (RPG, VaLiDRec, MaskGR) and **latent decoding** (L2D): there are no autoregressive per-level log-probs.
- **Prefill-only rankers** (Netflix GenRec, LinkedIn): there is no decoding.
- **Unconstrained decoders that keep raw logits** (NVIDIA's default scoring): bounds need normalized log-probs.

## 4. What would close the gaps

1. **A model with a wide first level.** OpenOneRec 1.7B (3 × 8,192, open weights, 15M items) is the obvious test. Measure how many first codes the top-K certificate needs on a real 8,192-code prior, and the hierarchical engine's certification rate at budgets of 64–512.
2. **More than 3 levels.** Extend `HierTrieEngine` to one budgeted pass per level and test 4-level SIDs (LC-Rec, GLIDE-style 4 × 256).
3. **Batched serving under a P99 budget.** Measure throughput at a fixed P99 with fallback reruns, against a batched beam engine, at 1k–3k uncached tokens: the deployed regime (L20/H100, 100–200 ms).
4. **eCPM-style value terms.** Run the GR4AD-like objective, log-probability plus log eCPM, inside the certified search, against SID beam followed by eCPM re-ranking.

## Sources

OneRec 2506.13695 / 2502.18965; OneRec-V2 2508.20900; OpenOneRec 2512.24762 and github.com/Kuaishou-OneRec/OpenOneRec;
GR4AD 2602.22732; GRACE 2608.00938; STATIC 2602.22647; xGR 2512.11529; UniSGR 2607.04068; NEZHA 2511.18793;
GLIDE 2603.17540 and the Spotify Engineering blog (2026-10-16); Netflix 2610.15864; NVIDIA recsys-examples
`examples/sid-gr-inference`; TIGER 2305.05065; LC-Rec 2311.09049; PinRec 2504.10507; MTServe 2604.22881; RelayGR 2601.01712.
