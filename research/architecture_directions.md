# Architectural directions for fast semantic-ID generative recommendation (MiniOneRec)

Scouting report, 2026-11-07. Sources were web search plus arXiv/HTML reads. "Not found" means my searches did not surface the work; it does not prove the work doesn't exist. This area moves weekly (one key paper below was posted 31 Oct 2026), so re-check arXiv before writing.

## TL;DR

- **Idea 1 (prefill-only + tiny SID head) is largely taken.** NEZHA (arXiv 2511.18793, WWW'26) runs one prefill with placeholder tokens, then an RNN head beam-searches 3×512 SIDs. It reports about 10× lower latency on Llama-1B and Qwen3-0.8B and is deployed at Taobao. SID-MLP (arXiv 2605.12617, May 2026) distills an AR teacher into per-digit MLP heads, including a 7B LC-Rec teacher. It recovers 83–97.5% of the teacher's accuracy at 17.9× end-to-end speed. Build this only as a baseline, or with a twist that is clearly different.
- **Biggest open lever: exact search instead of beam search.** MiniOneRec's catalogs are small (3,685 and 3,459 items) and its 3-level trie collapses after level 2. With tree attention, the unmodified model can score every valid `a` and every `(a,b)` in one forward pass. Prefix log-probs are upper bounds on full item scores, so a best-first, A*-style search can **certify the exact top-K in about 2 forward passes**, versus 5 sequential passes for beam-50. I found no prior work doing this for LLM-based GR. It needs no training, should give about 2–3× while being *exact*, and closes beam's search gap.
- **Second lever: depth.** I found no logit-lens, tuned-lens or early-exit study of LLM-based SID GR. "Where is the item decided across layers and SID levels?" is an open analysis question. Its answer motivates level-adaptive depth or a truncated backbone, which composes with exact search toward 5–10×.
- **Speculative decoding is capped at about 2.5× for 3-token SIDs.** Exact search dominates it.

---

## 0. Setup facts and a cost model

- **Verified MiniOneRec facts** (arXiv 2510.24431, 28 Oct 2025, technical report). SID vocabulary on Qwen2.5-Instruct; released checkpoints are Qwen2.5-1.5B-Instruct. RQ-VAE with L=3 and K=256. History is truncated to ≤10 items. Industrial has 3,685 items and 4,533 test users; Office has 3,459 items and 4,866 test users. The paper does not report an extra collision level.
- **Beam-50 cost.** Beam-50 with forced `\n` and EOS needs 1 prefill plus 4 *sequential* decode forwards: inputs `a`, `b`, `c`, `\n`. That drops to 3 if `c`+`\n` are fed together.
- **Batch-1 forwards are bandwidth-bound.** Qwen2.5-1.5B has 28 layers, d=1536 and tied embeddings with 151,936 vocab, so the LM head is about 15% of weights. In BF16 the weights are about 3.1 GB. An A6000 has 768 GB/s and roughly 150 TFLOPS dense BF16, so the roofline ridge is about 200 tokens per forward, and less in practice. **At batch 1, a forward with 50 query tokens costs about the same as one with 150–250.** Latency is roughly *(number of sequential forwards) × (bytes of weights read)*, not beam width.
- **Consequences:**
  1. Cutting sequential steps (5→2 or 5→1) is the main lever.
  2. Cutting bytes per forward (fewer layers, smaller model, weight-only quantization) is the second.
  3. Prompt compression or KV compression does almost nothing for batch-1 latency, because a 100-token prefill already costs about one weight read.
  4. The FP32 "exact" path goes compute-bound much earlier, so wide forwards cost relatively more there.
- **First action (15 min):** microbenchmark one forward with a shared 100-token prefix at 1, 50, 128, 256, 384, 512 and 1024 query tokens (BF16 and FP32). Also compute trie statistics: the number of used `a` codes, the number of distinct `(a,b)` prefixes, and the branching factor per level. Every speedup estimate below should be re-derived from these numbers. SID-MLP measured TIGER branching of 256 → ~38 → ~2.2 → ~1.2 per digit. With 3.6k items and 256² possible `(a,b)` pairs, expect `(a,b)` to almost identify the item, so level `c` is close to forced.

---

## 1. Prefill-only GR / distilled SID head

**Closest prior work**

| Work | ID / date / venue | What it does |
|---|---|---|
| **NEZHA** | 2511.18793, Nov 2025 (v2 Feb 2026), WWW'26 | One prefill over context plus L placeholder tokens; RNN draft head beam-searches L=3 (512 codes); hash-set *validity* check only, with no LLM verification. Trained by joint SFT of LLM and head, not distillation. Llama-1B: 74.2→7.0 ms; Qwen3-0.8B: 47.7→4.4 ms on Beauty, with H@10 roughly at parity. In production since Oct 2025. |
| **SID-MLP** | 2605.12617, 12 May 2026 (Guo, Hou, Ju, Shah, McAuley) | Frozen teacher context computed once; per-digit MLP heads take `[z; e(c1..c_{t-1})]`; loss is KL+CE distillation; trie-masked beam B=50. TIGER: NDCG parity at 8.74× throughput on an A6000. **LC-Rec 7B appendix:** attention readout over layer-8 KV with a last-layer query, 44.1M params, 83.3–97.5% recovery, 17.9× end-to-end. In their reimplementation, NEZHA recovers 65–82.7%. A teacher-for-first-m-digits variant ("m-mode", TIGER) loses 0.9% NDCG at m=1. |
| OneRec-V2 "lazy decoder" | 2508.20900, Aug 2025 | Context becomes static KV with no K/V projections. Cuts compute 94% versus OneRec-V1's encoder-decoder. Still autoregressive. |
| Parallel / NAR SIDs | RPG 2506.05781 (KDD'25); SETRec 2502.10833; LLaDA-Rec 2511.06254; MaskGR 2511.23021; DiffGRM 2510.21805; SPRINT 2609.34306 (Sep 2026, single pass, 8.4–10× faster than the next-fastest method); OneGR (ACM 2026, DOI 10.1145/3805712.3809884: one-pass position-wise scores plus WFST A*; full text not accessible) | Predict all SID tokens at once, or by diffusion; usually needs retraining and a different SID design. |
| Dense or catalog heads on an LLM (drop SID decoding) | DiGR 2602.07847 (Qwen3-4B user vector · mean-pooled SID embeddings; 0.074 s vs 0.683 s for beam-50 with *higher* NDCG); Google 2509.03746 (one token per item, two-level softmax, 5–14×); Netflix GenRec 2608.10257 (prefill-only with catalog head); CQG 2610.12977; LIGER 2411.18814 | With 3.5k items, a catalog softmax head on the prefill state is trivial and is a **mandatory baseline**. |
| General draft heads | Medusa 2401.10774; EAGLE 2401.15077 / EAGLE-3 2503.01840; MTP (Gloeckle) 2404.19737 | NEZHA argues that plain MTP heads fit SID structure poorly. |

**What would be new for us:** Not much. Defensible twists:
- Keep the *exact* full-model P(a) from the prefill and have the head model only P(b|a,user). For MiniOneRec, `c` is nearly forced.
- Distill at the *item level*: the teacher's full catalog distribution, which Idea 3 makes cheap to compute, rather than per-token teacher forcing.
- Run the first test of these heads on a **GRPO-trained** teacher.
- Score exhaustively with the head instead of beam search.

Together these could push recovery above SID-MLP's 83–97.5% on an LLM teacher. That is an incremental result.

**Expected speedup:** one prefill plus about 1 ms of head ≈ 6–9 ms versus 56 ms, so **~6–9×**. That is the speed ceiling for anything that keeps the 1.5B prefill. **Quality risk:** medium; LLM-teacher recovery in prior work ranges 65–97%. **Compute:** cache prefill states once (~30 min), then train heads in ~1 GPU-hour per dataset.

**Minimal experiment:** freeze MiniOneRec (SFT and GRPO checkpoints). Train three heads: (i) a SID-MLP-style head for P(a), P(b|a), P(c|a,b); (ii) the same head with P(a) taken from the LM; (iii) a catalog softmax head (DiGR-style). Report HR/NDCG@{3,5,10}, teacher-recovery % and top-10 overlap with the teacher, at matched latency.

---

## 2. Early exit, layer skipping, self-speculative beam decoding

**Closest prior work**
- **No GR-specific early-exit or LayerSkip paper found.** NEZHA drafts from the last layer. SID-MLP reads layer-8 KV but still runs all 32 layers for its query.
- Evidence for depth redundancy in recommendation:
  - SLMRec (2405.17890, ICLR'25): an 8-layer student matches 24–32-layer LLM recommenders; embedding-based, not SID generation.
  - EARN (2507.00715, Jul 2025): "layer-wise attention sparsity inversion" (early layers dense, later layers redundant) and dual attention sinks in LLM-based GR; register tokens give up to 3.79×.
  - PruneRec (2507.07064).
  - Mid-layer representation advantage for recommendation (2604.18146), contested by IMFuse (2607.27002), where the final layer is the best single layer.
- General methods: LayerSkip (2404.16710, ACL'24); Draft&Verify (2309.08168); SpecEE (2504.08850); tuned lens (2303.08112); "Reassessing Layer Pruning in LLMs" (ICLR'26): dropping the last layers plus fine-tuning the LM head and the last few layers works well.

**What is new for us:** a *per-SID-level* depth analysis and policy. The `a` step is a 256-way choice and carries the user signal. The `b` step chooses among about 14 valid children per `a` (3.6k items / ≤256 codes). The `c` step is nearly forced. A natural design is **level-adaptive depth**: full depth for the prompt and P(a), and an exit at layer L_b for the `a`-node tokens that produce P(b|a). In one tree forward, layers 1..L_b run on prompt plus `a`-nodes, and layers L_b+1..28 run on prompt tokens only.

**Self-speculative beam decoding** (draft with early layers, verify with the full model) only helps if verification collapses several sequential steps into one forward. For 3 SID tokens the best case is 5→2 full forwards plus draft cost, so ≈1.6–2.5×, and Idea 3 gets that without a drafter. **Deprioritize** self-speculation as a contribution.

**Expected speedup:**
- Level-adaptive depth alone: 1.3–2×.
- Truncating the whole model to L'≈14 (distilled): about 2× per forward.
- Either one combined with Idea 3: **~4–8×**.

**Quality risk:** medium; depends on what the lens analysis shows. **Compute:** tuned-lens translators for 28 layers in under 1 GPU-hour. Exit-head or truncated-model distillation takes 2–6 GPU-hours per (L', dataset).

**Minimal experiment:** see Idea 6 for the analysis. Then fine-tune a LayerSkip-style early-exit loss at L ∈ {10, 14, 18, 22}, with KL to the teacher's item-level distribution. Measure HR/NDCG and teacher top-10 overlap against latency.

---

## 3. Exhaustive exact trie scoring via tree attention (no beam)

**Closest prior work**
- **"Where Beam Search Loses Top-Ranked Items: First-Step Coverage in GR"**, arXiv 2611.01072 (31 Oct 2026, Ant Group). Uses an *offline* exhaustive exact-catalog reference for TIGER only. Every pruned exact top-10 path was lost at step 1. Beam-50 Search-Recall@10 ranges 0.88–0.99. Front-loaded widths [100, 30, 20, 10] reproduce Beam-100's top-10 at Beam-50's budget. The latency saving is only about 6%, and there is no tree attention and no LLM.
- **Item-Supported Decoding**, 2607.24995 (Jul 2026). In TIGER, only 29.9% of plausible targets survive SID filtering. An external item ranking protects prefixes; NDCG@10 gains reach 31%.
- **DiGR** (2602.07847) reports naive exhaustive AR scoring at **5.05 s/user** (Qwen3-4B, A100), versus 0.68 s for beam-50.
- Theory: Wu et al. 2504.09935 (SIGIR'25) on constrained-decoding error; PAG 2404.14600 (SIGIR'24), which plans with simultaneous decoding, reproduced in 2604.23396; BONSAI 2607.16633 on trie branching.
- Tree-masked one-pass scoring in recommendation: DICE 2610.34124 (≤100 text candidates; 3–6× over generation); InvariRank 2604.27599; AtSpeed (tree-attention verification).
- Exact or best-first search in NMT: Stahlberg & Byrne 1908.10090 (EMNLP'19), which found search errors that turned out to be useful; Best-First Beam Search 2007.03909 (TACL'20); Meister et al. 2010.02650 (EMNLP'20).

**What is new for us** (I found no prior work for LLM-based SID GR): **certified exact top-K in about 2 forwards.**

1. **Forward 1:** the prompt plus every valid `a` code as tree nodes. Each node attends to the prompt and itself, and all share position p. This yields exact log P(a) and log P(b|a) for every valid `(a,b)`, about 100 + ≤256 tokens.
2. Every remaining term (c, `\n`, EOS) is a log-prob ≤ 0, so U(a,b) = log P(a) + log P(b|a) is an **admissible upper bound**.
3. **Forward 2:** the top-M prefixes by U, each with its path `[b, c_i, \n_i]`. Paths are mostly unique because `c` is nearly forced. This gives exact full scores.
4. If the K-th best exact score ≥ max U over unexpanded prefixes, the top-K is **provably the global optimum** under the original full-vocab objective. Otherwise run one more round.

This removes beam search error, keeps MiniOneRec's scoring rule unchanged, and gives the first measurement of the search gap for an LLM-based GR.

Naive exhaustive scoring with the suffix needs about 256 + 3.4k + 3.6k + 3.6k ≈ 11k tree tokens. That is compute-bound, about 2–4× *slower* than beam, so use it only as the offline reference.

**Expected speedup** (BF16, batch 1): forward 1 ≈ 1.5–2× a decode forward, forward 2 (M≈64–100) ≈ 1.5×, so roughly 20–25 ms versus 56 ms, **~2–3×, exact**. If Idea 5 shows the suffix terms can be dropped or trie-renormalized, forward 2 is needed only for ambiguous `(a,b)` prefixes, giving **~4–5×**. FP32: about 2×.

**Caveat:** this uses more tokens per user than beam-50. Throughput at large batch may get worse unless combined with Idea 2 or 5. Report both regimes.

**Quality risk:** low. The output is the model's own argmax; 2611.01072 and ISD suggest exact search is neutral to positive for GR. **Compute:** inference only; offline references for about 9.4k test users take about 30 min.

**Minimal experiment:**
- Implement the tree mask with an HF 4D mask or FlashInfer custom mask, or as "shared-prefix KV plus a 256-wide batch".
- Compare beam-{10, 50, 100}, front-loaded beam, two-round certified search and offline exhaustive.
- Metrics: search-error rate (users whose beam top-10 differs from the exact top-10), HR/NDCG, certification rate after round 2, and p50/p90 latency at batch 1, plus throughput at batch 32.

---

## 4. Speculative decoding for beam-search GR

**Prior work**
- **AtSpeed** (2410.05165, ICLR'25): LC-Rec LLaMA-7B target, LLaMA-68M drafter, 4-token IDs, γ=4, draft beam 40, tree-attention verification. **1.97× (Beauty) and 1.83× (Games) with identical top-K** under strict verification; about 2.1× average and up to 2.5× relaxed. Measured on an A5000.
- **SpecGR** (2410.02939): about 1.7× with standard SIDs, about 5× with 16-digit SIDs.
- **PAD-Rec** (2604.27747, Apr 2026): position-aware drafter, up to 3.1× for list-wise multi-item generation.
- **NEZHA** "verifies" only that the SID is valid, so it is not lossless with respect to the LLM.

**Does it apply to 3 tokens?** Barely. The ceiling is 5 sequential forwards → 2 (prefill + 1 verify), about 2.5×, *minus* draft cost and rejections. Gains in prior work come from longer outputs (4–16 tokens or multi-item lists). Idea 3 is in effect self-speculation in which the model's own exact P(a), P(b|a) is the draft and an exact bound is the verifier. It reaches the same 2-forward floor without a drafter and is exact against the *global* optimum rather than against beam. **Not recommended** as a separate contribution; cite it as related work.

---

## 5. Scoring-objective simplifications

**Prior work**
- **FlashRec** is not a paper. It is a GitHub engine (sohu-mptc/FlashRec, 2026) that evaluates `lm_head` only over the SID token range, which implies SID-restricted normalization. It uses CUDA graphs and FP8 and supports beams of 50–512+. Its only quality check is beam-set overlap with HF: 86–92% in BF16 on OneRec-1.7B. On-load FP8 drops overlap to 70–80%, a useful caution about quantization.
- GENRE (2010.00904) masks log-probs *without* renormalizing.
- Wu et al. 2504.09935 gives KL lower bounds for constrained decoding.
- xGR (2512.11529) uses valid-path masks and early termination. It reports at least 3.49× throughput as a serving system.
- **No paper found that measures the ranking effect of dropping the forced `\n`/EOS terms, or of full-vocab versus codebook versus trie-renormalized scoring.**

**What is new for us:** a careful 4×2 study. Scoring rules: full-vocab (current); SID-restricted; trie-renormalized (locally normalized, with possible label bias toward sparse subtrees); and `(a,b)`-only. Search: beam versus exact (Idea 3). Without exact search, search error and scoring error are confounded. This decides how many forwards are needed:

| Scoring | Forwards |
|---|---|
| Full + suffix | 5 |
| No suffix | 3 |
| Trie-renormalized, `c` unique | 2, or 1 with tree attention |

Restricting the LM head also saves about 15% of bytes per forward.

**Expected speedup:** 1.15× (LM head) to 1.6× (drop suffix) on its own. It *enables* the 4–5× version of Idea 3. **Quality risk:** unknown and could go either way. The suffix probabilities act as a learned validity and confidence term, which is precisely what to measure. **Compute:** inference only, under 1 hour. A small analysis paper or section, not a headline.

---

## 6. Interpretability: where is the item decided?

**Existing 2024–2026 work** (none is a layer-wise mechanistic study of an LLM-based SID recommender):
- **Representation and search analyses:** 2607.24995 (SID representation → selection); Latte 2605.06331 (AR SID expressiveness limits, with near-tree items getting similar probabilities); 2611.01072 (first-step losses); SID-MLP §2.2 (difficulty concentrated at digit 1); PrefixMem 2606.00324 (SID token meaning is prefix-dependent; Qwen3-0.6B+PrefixMem beats Qwen3-4B on deepest-level SID accuracy).
- **Attention and layer redundancy:** EARN 2507.00715 (attention sparsity inversion and sinks in LLM GR); SLMRec (layer redundancy).
- **Non-LLM or adjacent recommenders:** SAEs for sequential recommenders 2507.12202 (RecSys'25); RecSAE 2411.06112; monosemantic concepts 2511.18024; "The Many Faces of One Head" (NeurIPS'26 affinity-workshop poster; DLA, patching and ablation on SASRec and GAT-Rec, finding heads near-identical).
- **IR analogs:** Probing Ranking LLMs 2410.18527; "How do LLMs understand relevance" 2504.07898.

**Gap:** no logit-lens, tuned-lens, activation-patching or attention-knockout study of a decoder-only LLM generating SIDs. That makes this a cheap, novel analysis section that directly motivates Ideas 2 and 3.

**Minimal experiment** (1–2 days, <2 GPU-hours):
- **Tuned-lens curves.** Per layer ℓ and SID level, measure top-1/top-10 agreement and KL to the final distribution, restricted to valid children, plus HR@10 if decoding exited at ℓ.
- **History-attribution knockout.** Block attention from the response position to history item *j*'s SID tokens. Compare recency, last item, and level-a vs level-b vs level-c tokens.
- **Activation patching across users** sharing the same `a` code, to separate "which branch" from "which leaf".
- **SFT vs GRPO checkpoints.** Does RL move decisions earlier or later?

Hypothesis to test: level `a` settles in middle layers, and `b|a` settles earlier given the `a` token.

---

## 7. Other levers

| Lever | Prior art | Batch-1 value | Notes |
|---|---|---|---|
| Smaller backbone + item-level KD (1.5B→0.5B) | SLMRec; MiniOneRec's own 0.5–7B scaling; PrefixMem shows small LLMs + better SID encoders can win | ~2.5–3× per forward | Strong "boring" baseline that must be compared against depth truncation. |
| Weight-only INT4/INT8 (Ampere: Marlin/AWQ) | general; FlashRec FP8 overlap drop | ~1.5–2.5× | Orthogonal, not architectural. Report as a baseline. FP8 tensor cores need the L40S, not the A6000. |
| History compression / user tokens | EARN (3.79×); User-LLM; Netflix GenRec (cost ∝ context, 5k→1.7k tokens ≈ ⅓ cost) | ~none at a 100-token prompt | Matters only for throughput or long histories. |
| KV compression, linear attention, MoE | — | negligible / infeasible in a week | Skip. |
| Better identifiers (PrefixMem, Latte, BONSAI) | 2606.00324, 2605.06331, 2607.16633 | quality, not speed | Could recover quality lost to truncation. |

---

## Ranked recommendation (overnight to one week)

**1. Certified exact top-K inference ("beam-free GR"). Do first; overnight to 2 days, no training.**
- *Contribution:* tree-attention scoring of all `a`→`b` transitions in one forward, plus admissible log-prob bounds, gives the provably exact top-K in about 2 forwards. Also yields the first beam search-error measurement for an LLM-based GR, on SFT and GRPO checkpoints.
- *Expected:* ~2–3× over your 56 ms BF16 engine with *equal or better* HR/NDCG; ~4–5× if the suffix study (Idea 5, run in the same week) allows trie-renormalized scoring.
- *Novelty:* medium-high. Closest is 2611.01072 (TIGER, offline exhaustive, beam reallocation).
- *Risk:* the speedup depends on your forward-cost curve. Run the microbenchmark first.

**2. Layer-wise decision analysis → level-adaptive and truncated depth with item-level distillation. 3–5 days, ~10–30 GPU-hours.**
- *Contribution:* the first tuned-lens and attention-knockout analysis of an LLM SID recommender (Idea 6). This sets the depth policy. Exit heads or a truncated backbone are distilled with KL against the teacher's exact *item-level* distribution, which method 1 makes cheap to compute.
- *Expected:* combined with method 1, **~5–10×** at small, quantified quality loss.
- *Novelty:* medium. Depth redundancy is known for embedding-based LLM recommendation (SLMRec, EARN), not for SID generation or per-level decisions.
- *Must compare against:* a distilled 0.5B model and INT4.

**3. Prefill-only heads as the lossy speed frontier (baseline-grade; 1 day, ~2–4 GPU-hours).**
- *What to build:* a SID-MLP-style head, a "LM P(a) + head for b|a" variant, and a DiGR-style catalog softmax head on the frozen prefill.
- *Why:* reviewers will ask for these. They bound what any method keeping the 1.5B prefill can achieve (~6–9×).
- *Headline only if:* it beats SID-MLP's 83–97.5% LLM-teacher recovery at matched latency.

**Suggested paper framing:** *"Beam search is unnecessary for LLM-based generative recommenders: exact, certified and depth-adaptive inference."* The contributions are the cost model, certified exact search, the layer-wise analysis, adaptive depth, and the scoring-rule study. Evaluate on Industrial and Office with HR/NDCG@{3,5,10}, report search-error rate and teacher overlap, and give latency at batch 1 and batch 32 in BF16 and FP32.
