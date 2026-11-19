# Literature sweep 2: inference efficiency for SID / generative recommendation (delta over `lit_ledger.md`)

Sweep date: 2026-11-08. Scope: 2023 – Nov 2026, with emphasis on Jul–Nov 2026 arXiv.

**What this file adds.** Only papers that are *not* rows in `research/lit_ledger.md`. A few (2611.01072, SPRINT, BONSAI) are already mentioned in `paper_draft.md` or `architecture_directions.md`; they are listed here because the ledger lacks them and some details matter.

**Method.**
- arXiv API full-text-abstract queries sorted by submission date (`abs:"generative recommendation" AND (latency|speedup|decoding)`, `abs:"semantic ID(s)" AND (beam|latency|decoding)`, `abs:"generative retrieval" AND (beam|constrained decoding)`, `abs:speculative`, `abs:"KV cache"`, `abs:"early exit"`, `abs:quantization`, `abs:renormaliz*`, `abs:"upper bound"`, `abs:exhaustive`, …); about 600 hits screened by title, about 120 abstracts read.
- WebSearch for venue lists (KDD/RecSys/SIGIR/IJCAI 2026), industrial reports, the vLLM RFC, and general-LLM analogs of late entry.
- WebFetch of the arXiv **HTML full text** for every paper rated *direct* or close *adjacent*.

**Verification key** (last column of each table):
- **H**: details checked in the arXiv HTML full text.
- **A**: arXiv abstract or API metadata only.
- **S**: secondary source only (summary site or search snippet). Treat as unverified.

**Threat key.** Each paper is rated against our three mechanisms. **M1** is certified exact top-K (branch-and-bound plus tree attention). **M2** is trie-normalized scoring. **M3** is late entry (frozen model, bridged lower layers). Ratings: **direct** means it claims the same mechanism or result. **adjacent** means a related idea that must be cited and differentiated. **none** means no overlap.

Total papers checked beyond the ledger: **~75** (~65 GR/recsys-specific, plus 5 general-LLM analogs of late entry and a few quantization and position papers).

---

## 0. Top-line findings

1. **M1 now has a direct prior.** *Beyond the Beam* (BB, arXiv 2609.33745, 2026-09-27) states the same certificate.
   - Bound: an unevaluated item's score is at most its evaluated prefix sum, because log-probs are ≤ 0 (plus a known item-level correction).
   - Stopping rule: stop when the K-th evaluated score exceeds the best unevaluated bound, which certifies global top-K (their Prop. 2).
   - **Differences from ours:**
     - BB is *post-hoc completion after a full beam search* (beam 40), teacher-forcing ≤80 extra items in batches of 20.
     - It certifies only 4.5–77.6% of queries (top-20).
     - It uses **full-vocabulary log-probs with no allowed-child renormalization** (stated explicitly), so its bounds are loose.
     - It uses no tree attention and runs on a small T5 / LC-Rec (RTX 3090, ~10–12 ms per query).
     - Its goal is catalog-expansion quality (new items), not speed.
   - **Consequence:** our draft's sentence "no prior work … gives a per-request optimality certificate" is now false and must be changed.
2. **M2 has a direct partial prior.** *ZeroGR* (arXiv 2611.04715, 2026-11-03, HKU + Kuaishou) runs log-softmax **over valid trie children only** ("filtering-first"). It explicitly contrasts this with full-codebook normalization ("scoring-first"), saying the two can rank differently.
   - It measures the effect on MovieLens-1M as negligible: 99.85–99.90% list overlap and near-identical Recall@{1,10,50,128}.
   - The motivation is compute (skip invalid logits), not search tightness. The quality conclusion is the *opposite* of ours: we see +1.2 pt HR@10 on an LLM over a 152k vocabulary.
   - **Consequence:** we can no longer say normalization is unmeasured. We can say it was measured as negligible on a small non-LLM model, while it matters for LLM recommenders whose softmax leaks mass to non-SID tokens.
3. **M3's motivation is now crowded, but the mechanism is not.** "Deeper SID levels are easier, so spend less compute on them" is now published in several forms:
   - BRIDGE (2611.07425): trained recurrent loops with a (4,2,1) per-level budget.
   - IBA (2607.12425): information gain falls with SID position.
   - SmartGR (2608.02048), LoopGR (2610.34157), PFA (2610.39273): later steps attend to fewer tokens.
   - 2608.21012: drops a level entirely.
   - General-LLM analogs: DMTD (2510.11958), where later tokens run only the late layers after fine-tuning, and D³ (2503.08524).

   None of these is post-hoc on a frozen LLM with a child-entry state predicted from the parent's state at the same depth, and none composes with a search certificate. That remains our defensible core.
4. **New batch-1 reference points for ~1.5–1.7B LLM recommenders.**
   - Merge & Branch (2610.37973): Qwen3-1.7B, beam 100, B=1, H200, **77–84 ms** with a tree KV cache.
   - LatentRec (2611.07866): Qwen3-1.7B, beam 50, B=1, H20, **590 ms**.

   Our 2.9 ms certified top-10 on an H100 is 25–200× below these (different GPUs; same model class).

---

## 1. Paper digest

### 1A. Exact search, certificates, beam-error analyses, one-pass candidate scoring

| Paper | ID / venue | Date | Mechanism | Headline numbers | M1 | M2 | M3 | Ver. |
|---|---|---|---|---|---|---|---|---|
| **Beyond the Beam: Constructive Repair and Candidate Completion for GR** | arXiv 2609.33745 | 2026-09-27 | Two parts. (1) Identifier-assignment repair via integral flow, for catalog expansion. (2) **Budgeted completion after beam search**: score F(i) = Σ full-vocab log p + d_i (d_i is a ridge collaborative correction). Unevaluated items get bound U_i = (evaluated-prefix log-prob) + d_i, valid because each token log-prob ≤ 0. Each round teacher-forces the β=20 highest-bound items, up to 80 extra. **Certificate: stop when t_K > max_{i∉E} U_i** (Prop. 2), checked with a 1e-3 numerical allowance that is "not a formal floating-point error bound". | Recall@10 +15.5–46.3%, NDCG@10 +15.2–44.4% over the best GR baseline (new-item setting).<br>Top-20 certification rate: **77.6% Beauty, 4.45% Tools, 61.5% Toys** (bound priority).<br>~10.2–12.0 ms per query batched (RTX 3090, T5, beam 40 plus completion). Also run on LC-Rec. Catalog ~11.8k (Beauty).<br>States "No allowed-child renormalization". | **direct** | adjacent (opposite choice) | none | H |
| Where Beam Search Loses Top-Ranked Items: First-Step Coverage in GR | arXiv 2611.01072 (Ant Group) | 2026-10-31 | **Offline exhaustive** exact-catalog reference for TIGER. Every pruned exact top-10 path is lost at step 1. Reallocating width to the first step ([100,30,20,10], budget 151) reproduces Beam-100's ordered top-10. | Search Recall@10 +0.96–10.90 pt at Beam-50 budget.<br>Decoder latency −6.35–6.39%.<br>**Forward/cache management is 76.8–79.6% of Beam-100 wall time**, so less search work barely cuts runtime.<br>FP32, PPU device. Normalization not stated. | adjacent (motivates scoring all first codes; the reference is offline only) | none | none | H |
| OneChord: Distill What You Rank, Recommend in One Pass | arXiv 2610.25805 (UCAS) | 2026-10-19 | One parallel pass. A **catalog-normalized** mixture of a SID branch (products of per-position softmaxes, renormalized over legal SIDs) and an item-matching branch.<br>Prop. 3.3: if the K/K+1 margin exceeds √(2·KL), the direct and refined top-K sets coincide. | R@5 / N@5 +6.30% / +7.12% (6 Amazon sets).<br>1.04–1.18× faster than SPRINT.<br>Non-LLM (d=256, 4 decoder layers). | adjacent (top-K preservation condition, not a search certificate) | adjacent (renormalizes over the legal catalog) | none | H |
| TCBSA: Test-Time Collaborative Beam Search Adaptation | arXiv 2610.32098 | 2026-10-23 | Training-free. Discriminative score margins become potentials on shared semantic edges, added before beam selection. Gives a sufficient condition for target-prefix survival under global beam competition. | Better accuracy/latency than widening the native beam (full-catalog evaluation). | adjacent | adjacent (score reshaping) | none | A |
| V-STAR: Value-Guided Structured Sampling | arXiv 2602.10699 (Tencent) | 2026-02-11 | Value-guided efficient decoding (VED): build a shallow beam tree, then spend budget deepening "decisive", high-value nodes "without exhaustive tree search". Sibling-GRPO for RL. | Better accuracy and diversity under latency limits (no ms in the abstract). | adjacent (budgeted best-first, no guarantee) | none | none | A |
| PROMISE: PRMs Unlock Test-Time Scaling in GR | arXiv 2601.04674 (Kuaishou) | 2026-01-08 | A lightweight process reward model guides beam pruning. A smaller model with wider search matches larger models. | Offline and online gains; no latency in the abstract. | adjacent | none | none | A |
| DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify | arXiv 2607.08365; **IJCAI'26** (Alibaba HUJING) | 2026-07-09 | A vector-search draft of N candidates. **All N are verified in one parallel forward** over a broadcast shared-prefix KV ("Broadcasted Prefix Caching"), with a fused likelihood + similarity score. | **≈70 ms** vs ≈130 ms cascade vs **≈3 s** for pure AR (TIGER-like); industrial video search with >100M items.<br>Backbone unnamed, N unstated, mask unstated. | adjacent (one-forward multi-candidate scoring with a shared prefix, but over a drafted set and with no certificate) | none | none | H |
| Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval | arXiv 2610.08716 | 2026-10-06 | Controlled AR vs masked/block diffusion comparison. **One-pass scoring** reads a fully masked identifier once and scores every document by its codes' marginals. | Decoding alone moves diffusion Hit@1 by 6.6–13.7 pt. AR still leads at Hit@1, and on NQ320K the lead comes "from the model, not beam search". | adjacent (exhaustive scoring with marginals, which DiGR shows cannot preserve AR ranking) | none | none | A |
| Merge and Branch: Tree-Structured SFT of LLMs for Recommendation | arXiv 2610.37973 (Georgia Tech) | 2026-10-27 | Tree FlashAttention (training kernel) for one history feeding many target branches. A **tree-structured KV cache** for constrained beam decoding: the prefix is stored once, with per-hypothesis SID states and a GPU trie. | **Qwen3-1.7B, H200, B=1, beam 100: 77.1 / 76.4 / 83.9 ms** at context 64 / 1024 / 2048, vs 86.0 / 91.1 / 139.9 ms with duplicated KV. Memory −86% at 2048. | adjacent (tree attention in an LLM recommender, but for beam search) | none | none | H |
| PATH: History-Conditioned Joint-Prefix Alignment | arXiv 2608.29179 | 2026-08-29 | Most missed targets are pruned within the first two steps. KL-aligns two-token prefix predictions to corpus transition statistics, then **PMI calibration** reranks completed candidates. | Higher full-SID survival on Beauty, Instruments and Yelp. | adjacent | adjacent (score calibration) | none | A |

### 1B. Normalization and constrained-decoding systems

| Paper | ID / venue | Date | Mechanism | Headline numbers | M1 | M2 | M3 | Ver. |
|---|---|---|---|---|---|---|---|---|
| **ZeroGR: Filtering-First Constrained Decoding for GR** | arXiv 2611.04715 (HKU, PKU, HUST, **Kuaishou**) | 2026-11-03 | Look up valid next tokens first (a step-sliced filtering index: sparse lists or dense successor tables). Then either a **selective projection** (dot products only for valid tokens' head rows) or a full projection with −∞ masking, chosen by warm-up latency. Then **log-softmax over valid slots only**. Also shared-slot execution and streaming index rebuild. | **Up to 9.9×** on the filtering + scoring + selection pipeline (model inference excluded). 9.1× over Unconstrained on "Search" (FP16, batch 32). Up to 205× over a CPU trie. GPU index up to 74.5% smaller than STATIC. Index rebuild 278× faster.<br>**Quality: filtering-first vs scoring-first overlap 99.85–99.90%** at top-{1,10,50,128}, near-identical Recall (MovieLens-1M only; model unnamed).<br>RTX 4090D, beam 128. | none | **direct** (same restricted normalization, measured; opposite conclusion) | none | H |
| PinGER: Token-Efficient Generative Retrieval with SIDs | arXiv 2610.33435 (Pinterest) | 2026-10-24 | Asymmetric mask so history items cost one token each. Serving: **cascade decode** (beams folded into the query dimension, shared-prefix and suffix attention merged by log-sum-exp in one fused launch). STATIC-style GPU trie (dense first 2 levels, CSR deeper). **"Codes outside the allowed set are masked to −∞ after the log-softmax"** (full per-level codebook normalization, K=2048, L=5). | Online W=60–100.<br>L40S: model p99 12–23 ms; end-to-end median 48–61 ms vs 79 ms (PinRec).<br>Attention 10.9–32× faster than paged attention (A10G micro-benchmark).<br>Trie 168 MiB for 10M items; 0.30 ms at W=100.<br>Saves +0.62%. | none | adjacent (opposite choice, documented) | none | H |
| GR under Real-Time Supply Constraints in Local Commerce (RTP-LLM) | arXiv 2611.06323 | 2026-11-04 | A request-time supply snapshot is compiled to a CSR prefix index, overlapped with prefill and shared across beams. **"Legal expansions retain original-model log probabilities"** (no renormalization). | +0.6 ms for 30k entries. Filtered merchants 170/299 → 29/299. | none | adjacent (opposite choice) | none | A |
| On the Equivalence Between AR-NTP and Full-Item-Vocabulary MLE (short note) | arXiv 2604.15739 (Kuaishou) | 2026-04-17 | Claims AR next-token prediction ≡ a softmax over all items, given a bijection and full Cartesian coverage (N = X^k). Each step normalizes over the full position codebook. | Theory only.<br>My reading: the proof holds for parallel tokenization. For cascaded (prefix-dependent) normalizers, identifying the product of normalizers with the item partition function is unproven. | none | adjacent (theory hook: a trie-normalized product is a proper item distribution) | none | H |
| PRISM (adaptive temperature by trie branching density) | arXiv 2601.16556 | 2026-01-23 | Sets the per-position softmax temperature from trie branching density. | Not checked. | none | adjacent | none | S |
| UAGR: Utility-Aware GR (utility-guided decoding) | arXiv 2611.02981 (Kuaishou) | 2026-11-02 | Adds multi-outcome utility estimates to the beam score to stop premature pruning of high-utility paths. | Deployed; watch time +0.206%. | adjacent (an augmented objective that B&B could certify) | adjacent | none | A |
| RGD: Reward Guided Decoding for GR | arXiv 2607.25344 (Kuaishou) | 2026-07-28 | Closed-form KL-regularized reward-tilted decoding distribution; a reward model acts as a test-time controller at each step. | Deployed (no numbers in the abstract). | adjacent | adjacent (per-step score reshaping) | none | A |
| GR4AD-X | arXiv 2610.38378 (Kuaishou) | 2026-10-28 | Multi-path SID assignment; "serving-efficient capacity scaling"; value-guided beam pruning plus discriminative rerank (UniVGD). | +3.07% revenue over GR4AD under the same latency and throughput constraints. | adjacent | none | none | A |
| Gryphon / Gryphon-v2 | arXiv 2606.08604 / 2608.06213 (Yandex) | 2026-06-07 / 08-06 | Item-level scoring head reuses encoder states and re-scores beam outputs, citing "sequence likelihood miscalibration". v2 distills a teacher ranker via rollouts. | +4.2% Recall@1000 vs beam-likelihood ranking of the same candidates. v2 replaces a 15+ generator cascade (+1.41% active users). | none | adjacent (sequence-score calibration) | none | A |
| GPR (value-guided trie-based beam search) | arXiv 2511.10138 (Tencent) | 2025-11-13 | One-model ads generator: HHD dual decoder, MTP training. "Value-Guided Trie-Based Beam Search" appears only in a figure. | Deployed in WeChat Channels ads. | adjacent | none | none | A + S |
| vLLM RFC #45815 "High Beam-Width GR Inference" | GitHub | opened 2026-06-16; closed "not planned" 2026-10-16 | Beam-aware scheduling plus shared-prefix beam attention for OpenOneRec/TIGER. A maintainer points to `neuralmagic/vllm-beamsearch-plugin`. | No numbers. | none | none | none | H (GitHub page) |

### 1C. Per-level compute allocation, depth, recurrence (late-entry neighbourhood)

| Paper | ID / venue | Date | Mechanism | Headline numbers | M1 | M2 | M3 | Ver. |
|---|---|---|---|---|---|---|---|---|
| **BRIDGE: Bridging the Semantic Quantization Gap for GR** | arXiv 2611.07425 (Kuaishou) | 2026-11-05 | Three parts: codebook-grounded latent reasoning, privileged EMA self-distillation, and **effort-aware recurrence (EAR)**. EAR is one shared 3-layer loop block over an expanding SID prefix with a **right-aligned per-level budget m=(4,2,1)**, "more iterations to early, high-entropy levels". A trained small transformer (d=512); Qwen2.5-1.5B is used only for item embeddings. | R@10 / N@10 +10.9% / +9.7% on average (Beauty, Sports, Toys).<br>Analytic FLOPs −22% vs a 9-layer decoder.<br>**~20% QPS over GR4AD on L20.**<br>eCPM +0.65% (1-week A/B, 5% traffic).<br>No ablation over the m schedule. | none | none | **adjacent** (same "deeper levels need less compute" premise; trained architecture, no frozen backbone, no state bridge) | H |
| IBA: Information-Gain Budget Allocation for latent reasoning | arXiv 2607.12425 | 2026-07-14 | Measures position-wise information gain: earlier SID positions carry more. Learns how to split latent refinement steps across positions. | Better accuracy/compute than fixed allocation (public sets). | none | none | adjacent (evidence for level-dependent compute) | A |
| LoopGR: Recurrent Depth for GR | arXiv 2610.34157 (Tencent + SYSU) | 2026-10-25 | Shared decoder plus context updater looped T times. **Recurrent trajectory self-distillation** trains a lower-loop student. The schedule is uniform across SID positions. | B=1: 16-loop teacher 211 ms → 8 loops 104 ms → 4 loops 49.8 ms (2.0× / 4.2×). Up to 52% fewer parameters (1.3M vs 2.7M). | none | none | adjacent (depth reduction by distillation, uniform per position) | H |
| LoopRec: Item-Level Adaptive Recursion for Sequential Rec | arXiv 2610.30128 (Cao, Tang, Zhou, Dai, Xu; code Elpmis117/LoopRec) | 2026-10-21 | A weight-shared block with a per-item exit gate. Depth is allocated per *history item* (not per SID level). Learned without depth labels. | Beats matched fixed-depth models on 4 backbone families. **Numbers not verified** (abstract has none). | none | none | adjacent (weak) | A |
| SmartGR: Hierarchy- and Beam-Aware Distillation | arXiv 2608.02048 | 2026-08-03 | OneRec-8B → OneRec-1.7B. Per-level KL weights are learned with a monotone prior ("deeper SID levels receive no smaller weights"). Beam-aware ranking distillation. | +8.6% quality. **2.39× average speedup, entirely from the smaller student.** | none | none | adjacent (weak) | H |
| PFA: Progressively Focused Attention | arXiv 2610.39273 (Alibaba Intl.) | 2026-10-29 | Full attention at the first SID step; then a cross-beam aggregated importance ranking, KV reordered contiguously, and a token budget shrinking geometrically (k=256, r=0.5) at later steps. | Industrial: 62 vs 90 ms per batch (1.45×; L20; beam 100; small encoder–decoder). Online 11 ms vs 15 ms with a 512-item history. | none | none | adjacent (makes deeper SID steps cheaper via attention width, not depth) | H |
| Static Multi-Level → Dynamic Single-Level Large Codebook | arXiv 2608.21012 (Kuaishou) | 2026-08-21 | Replaces [256,256,256] RQ codes with [1024 semantic + 512 hash disambiguation]: 3 AR steps become 2. Exposure-aware codebook refresh. | Estimated AR FLOPs −47.9–48.7%.<br>Single-card QPS +47.0% (decoder), +28.57% (LazyAR), +33.3% (MTP).<br>Online +0.792%. | none | none | adjacent (drops a level instead of cheapening it) | H |
| LESSRec: Level-Specific History Modeling | arXiv 2610.37286 | 2026-10-27 | At inference, the history tokens at the *target's* SID level are the most informative. Attention to history is restricted per level. LESSRec+ cuts encoder compute. | Efficiency analyses only (no numbers in the abstract). | none | none | adjacent (weak) | A |
| LASAR: Latent Adaptive Semantic-Aligned Reasoning | arXiv 2605.10207 | 2026-05-11 | Latent reasoning with a policy head choosing per-sample depth (on top of MiniOneRec). | Roughly halves latent steps; ~20× faster than explicit CoT. | none | none | none | A |
| LatentRec: Do Recommenders Need to Think in Words? | arXiv 2611.07866 | 2026-11-05 | Per-step head supervision during training; the latent module is discarded at inference. | **Qwen3-1.7B, beam 50, H20, B=1: 0.59 s per sample** vs 10.8 s for CoT SFT (Beauty). | none | none | none | H |

### 1D. Serving systems, kernels, KV and memory

| Paper | ID / venue | Date | Mechanism | Headline numbers | M1 | M2 | M3 | Ver. |
|---|---|---|---|---|---|---|---|---|
| OneLA: Linear-Attention Decoding at Large Beams | arXiv 2609.12399 (HKU + Kuaishou) | 2026-09-11 | For Gated-DeltaNet GR: one shared prompt state plus append-only divergent transition records per beam; an ancestry index; a fused kernel. | 1.54–2.46× end-to-end decode vs FullState/ReplaySSM paths. | adjacent (tree-structured sharing for recurrent state) | none | none | A |
| ReuseGR: Offline–Online Decoupled GR via Adaptive Prefix Reuse | arXiv 2611.04259 | 2026-11-03 | A 7B offline planner builds per-query SID prefix sets. A 0.5B online model completes them per user. ReuseCut picks the cache depth; Path KD; BudgetCut (OPD + GRPO) selects branches. | Mean online latency −38.9% vs all-online 0.5B; matches the 7B HR@100 with −57.7% latency. CTR +0.93% (14-day A/B). Search ads, 3.38M items. | adjacent (cross-request prefix reuse) | none | none | A |
| HBF KV cache for GR serving | arXiv 2609.07175 | 2026-09-07 | High-Bandwidth Flash for user-KV retention, with an admission-controlled LRU-K write-aware policy. | 3.8–4.7× throughput vs HBM-only (analytical). Flash lifetime ~1 → 6+ years. | none | none | none | A |
| UniPinRec | arXiv 2606.00422 (Pinterest) | 2026-05-29 | One model for retrieval (ANN) and ranking (cross-attention). **Cross-stage KV cache sharing.** | End-to-end latency −11.1%, QPS +63.6%, ~+1% engagement. | none | none | none | A |
| GBLA: Gated Bidirectional Linear Attention for GR | arXiv 2606.07317 (Yandex) | 2026-06-05 | Linear-time bidirectional encoder, interleaved 1:2 with self-attention. | Up to 8.2× single-layer speed vs FA3 at 32k context (H100). | none | none | none | A |
| GRP v0.1 Technical Report | arXiv 2609.36688 | 2026-09-29 | Encoder–decoder retrieval, ranking and reward model; mGRPO. | **Serving optimizations: −69% retrieval latency** (details not in the abstract). | none | none | none | A |
| FLAME: Serving System for Large-Scale GR | arXiv 2509.22681 | 2025-09-17 | CPU-GPU decoupling; Proximal Data Accelerator (PDA) memory module; TensorRT fused-kernel engine; dynamic stream orchestration. | PDA: 1.9× throughput / 1.7× latency. FKE: 4.6–6.1× speed. DSO: 1.3× throughput. | none | none | none | A |
| GEMs: Multi-Stream Decoder | arXiv 2602.13631 | 2026-02-14 | Recent, mid-term and lifecycle streams; offline–online compression; >100k interactions. | "Superior inference efficiency" (qualitative). | none | none | none | A |
| UniSGR-V2 / TacGR | arXiv 2610.16132 (Lazada) | 2026-10-12 | Target-side feature anticipation per level; a pyramid memory funnel cuts inference cost. | Orders +5.55%. | none | none | none | A |
| Rec2PM: Recurrent Preference Memory | arXiv 2602.11605 | 2026-02-12 | Compresses long history into persistent preference-memory tokens. | Not verified. | none | none | none | S |

### 1E. Fewer or parallel decode steps, decoder lightening, "no LM at serving"

| Paper | ID / venue | Date | Mechanism | Headline numbers | M1 | M2 | M3 | Ver. |
|---|---|---|---|---|---|---|---|---|
| SPRINT: Single-Step GR via Average Probability Velocity | arXiv 2609.34306 | 2026-09-28 | All token probabilities from one bidirectional forward, plus a dual-level flow contrastive objective. | **8.39–10.04×** faster than the second-fastest AR/NAR method; +7.77% accuracy. | adjacent (alternative to AR search) | none | none | A |
| FLASH: SimHash SIDs with Parallel Decoding | arXiv 2610.07402 | 2026-10-05 | Training-free SimHash IDs with parallel decoding and semantic alignment. | State of the art (no speed number in the abstract). | none | none | none | A |
| InforID: Adaptive Semantic Capacity Allocation | arXiv 2608.09685 | 2026-08-10 | Per-slot codebook sizes and effective ID length for parallel GR. | Better at equal capacity; one-step prediction. | none | none | none | A |
| HGenPush (Chained-MTP) | arXiv 2607.03362; **KDD'26** (Kuaishou) | 2026-07-03 | Lightweight MTP "discards the autoregressive paradigm" for push. | DAU +0.181%. **No latency numbers.** | none | none | none | A |
| Towards Generalizable and Efficient Large-Scale GR (Netflix) | arXiv 2605.23312 | 2026-05-22 | MTP for serving-latency alignment; sampled softmax; projected decoding head; scaling to a 1B backbone. | Production shadow test, MRR up. | none | none | none | A |
| GenCDSR: serial-parallel decoding | arXiv 2607.28659 (RecSys'26 code) | 2026-07-21 | Partially parallel decoding over hierarchical SIDs. | Latency −85.1%, accuracy +1.5%. | none | none | none | A |
| Atomic Creator Tokens (ACT) | arXiv 2611.00323 | 2026-10-30 | One vocabulary token per creator; inference is a direct top-k over logits (exact by construction). | Recall@15 1.84× vs SID; **~2.7× lower per-user latency.** | adjacent (exact top-K without search, via atomic IDs) | none | none | A |
| Bit-Factorized Identifiers | arXiv 2610.17467 | 2026-10-12 | DocID tokens scored as sums of bits from model representations. | Up to 7× lower latency than numeric docids. | none | none | none | A |
| SLIDE: Search-to-List Item Distillation | arXiv 2610.34764 | 2026-10-26 | An offline beam teacher's top-K SIDs are concatenated into a list target; the student emits the list in one greedy pass. | Beats the beam reference with stronger teachers (FORGE, Taobao). | adjacent (exact top-K could replace the beam teacher) | none | none | A |
| LIME-Rec ("Is serving-time LM inference necessary?") | arXiv 2608.01260 | 2026-08-02 | Non-LM fusion beats GRAM (an LM-based GR) on 12/12 dataset-metric pairs. | Recall@10 up to +15.2%. | positioning-relevant | none | none | A |
| CQ-SID (Tmall search) | arXiv 2605.14434 | 2026-05-14 | Category- and query-constrained cluster SIDs halve the beam; EG-GRPO. | Beam ×0.5. Production numbers (200+ QPS on 8 GPUs, ~40 ms) are **S** only. | none | none | none | A (+S) |
| IntRR / STAMP / PSG / PushDualGen | arXiv 2602.20704 / 2604.05329 / 2607.26427 / 2608.07989 | 2026 | IntRR: one input token per item. STAMP: training speed 1.23–1.38×. PSG: pair-token reranking 1.83×. PushDualGen: SID first, then a skippable copy. | — | none | none | none | A |
| Multi-Decoder OneRec; BARGE; SPAR; RoFiGR | arXiv 2607.26500 / 2607.21028 / 2610.27898 / 2610.39470 | 2026 | Quality-side decoding variants: multi-decoder constrained beam with quotas, dual-path decoding, prefix-balanced SIDs, level-adaptive distillation. | — | none | none | none | A |

### 1F. Training-side beam-awareness (motivates exact search)

| Paper | ID / venue | Date | Mechanism | M1 | M2 | M3 | Ver. |
|---|---|---|---|---|---|---|---|
| BEAR: Beam-Search-Aware Regularization | arXiv 2601.22925; **SIGIR'26** | 2026-01-30 | Each positive token must rank in the top-B at every step (a relaxed necessary condition for beam survival). | adjacent | none | none | A |
| APAO: Adaptive Prefix-Aware Optimization | arXiv 2603.02730 | 2026-03-03 | Prefix-level losses; worst-prefix focus. | adjacent | none | none | A |
| GRC: Reflect and Correct | arXiv 2602.23639 | 2026-02-27 | Generation, reflection, correction; entropy-guided reflection scheduling during beam search. Revenue +1.79%. | none | none | none | A |

### 1G. Quantization and compression (ledger gap f)

| Paper | ID | Date | Finding | Relevance | Ver. |
|---|---|---|---|---|---|
| Quantization Amplifies Determinism, Not Bias | arXiv 2609.07901 | 2026-09-07 | Qwen3-8B AWQ-INT4 narrows suggestions: same-brand recommendation probability +5.1 pt; first-token distribution more concentrated. No effect at 14B or 32B. | First evidence that **INT4 changes recommendation-like choices** at ~8B. Not a SID model, so the ledger's gap (f) for SID-GR ranking stays open. | A |
| Scaling Down, Serving Fast (LinkedIn) | arXiv 2502.14305 (EMNLP'25 Industry) | 2025-02 | Distillation, structured pruning and quantization of recommendation SLMs. | Adjacent to gap (f); not SID-GR. | A |

### 1H. General-LLM analogs of late entry (outside GR; cite in related work)

| Paper | ID | Mechanism | Relation to M3 | Ver. |
|---|---|---|---|---|
| **DMTD: Direct Multi-Token Decoding** | arXiv 2510.11958 | After one full pass, the next tokens of a cycle use **only the late layers** (Qwen3-4B: last 8 of 36). Skipped KV is refilled at the next cycle. **Fine-tuned end to end.** Cycle 3–4 gives ~2× at B=1, with 96–98% quality. | Closest in shape: later tokens skip the lower layers. It differs from ours: no bridge (late layers consume the token directly), a fine-tuned model, and KV refill. | H |
| D³: Position-Aware Depth Decay Decoding | arXiv 2503.08524 | Training-free. Later tokens keep ⌊L·α^i⌋ layers. 1.5× ops saved on Llama 7–70B. | Supports "later tokens need less depth". It truncates the top, not the bottom. | A |
| KV Prediction (Apple) | arXiv 2410.08391 | An auxiliary model predicts the base model's KV cache for the prompt. | Predicting a frozen base model's internal states from cheaper compute, the same family as our bridge. | A |
| SwiftKV (Snowflake) | arXiv 2410.03960 | Prompt tokens skip *later* layers: their KV comes from an earlier layer's output, with distillation. Prefill −25–50%. | The mirror image of late entry. | A |
| Probe and Skip (SPTS) | arXiv 2601.13155 | A low-rank proxy predicts token transformations to skip computation. | Adjacent (weak). | A |

### 1I. Unverified or unresolved

- **CoSID: Concept-Conditioned Semantic-ID Decoding for Efficient GR** (ACM DOI 10.1145/3773078.3841283, reportedly published Sept 2026). The ACM page returned 403. Search snippets describe "predict a concept once, run beam search in a lightweight KV-cached decoder", which matches Deco (2610.32980, already in the ledger). Very likely Deco's venue version. **Unverified.**
- PRISM's adaptive-temperature formula (S only); GPR's value-guided trie beam (figure only); Rec2PM numbers; LoopRec numbers; CQ-SID production QPS and latency.
- "Two-dimensional early exit" (arXiv 2604.18592): a search snippet claimed recommendation datasets, but the abstract says sentiment classification. **Not a GR paper.**

---

## 2. New batch-1 / small-batch latency reference points (beyond ledger §3)

| System | Model | HW | Beam / K | Batch | Latency | Source |
|---|---|---|---|---|---|---|
| Merge & Branch (tree KV) | Qwen3-1.7B (fine-tuned with token merger) | H200 | beam 100 | 1 | **77–84 ms** (ctx 64–2048); duplicated KV 86–140 ms | 2610.37973 Table 4 (H) |
| LatentRec | Qwen3-1.7B | H20 | beam 50 | 1 | **590 ms** per sample | 2611.07866 Table 3 (H) |
| LoopGR | ~1–3M-param recurrent | unstated | — | 1 | 211 → 104 → 50 ms (16 / 8 / 4 loops) | 2610.34157 App. A.2 (H) |
| Beyond the Beam | T5 (TIGER-style) | RTX 3090 | beam 40 + ≤80 completions | batched mean | ~10.2–12.0 ms per query | 2609.33745 (H) |
| DaV-Gen | unnamed PLM | unstated | N drafted candidates | — | ≈70 ms (AR baseline ≈3 s) | 2607.08365 (H) |
| PinGER | 768-d decoder, L=5 SID | L40S | W=30–400 | ~200 QPS per host | model p99 12–23 ms | 2610.33435 (H) |
| PFA | 2+2-layer enc–dec | L20 | beam 100 | 512 per GPU | 62 ms per batch; online 11 ms | 2610.39273 (H) |
| **RapidTinyRec (ours)** | Qwen2-1.5B, 28 layers | H100, FP16 | certified top-10 | 1 | **2.9 ms** | PLAN.md |

The honest comparison is Merge & Branch: same model class (1.7B Qwen), beam 100 vs our certified top-10, and an H200, which is at least as fast as our H100. That puts us about 25× lower. Their number is for beam-100 output (K=100); ours certifies top-10, so add our certified top-500 at 3.8 ms for a matched-K claim.

---

## 3. Gaps and the five most promising unexplored optimizations (for our setup)

1. **Boundary-verified late entry (late entry as a draft, full depth as the verifier).**
   - Today the certificate covers only the *bridged* objective (draft §10 limitation).
   - Proposal: calibrate δ_c on validation, either as the max or as a conformal (1−α) quantile of |bridged − full-depth| third-code log-prob.
   - After certification, re-score at full depth only the leaves whose bridged score is within δ_c of the K-th score. Merge those rows into a tail forward, or skip it when the set is empty.
   - Result: the certificate refers to the unmodified model, with an explicit failure probability α.
   - Nobody does this: BB certifies exact scores only; DMTD, D³ and BRIDGE have no verification step.
2. **Certified top-K for value-augmented objectives via subtree-max annotations.**
   - Industry adds item-level or edge terms to beam scores: BB's d_i, UAGR utility, RGD reward tilt, GR4AD-X value-guided pruning, TCBSA potentials. All of them cite "premature pruning of high-value candidates" and patch it heuristically.
   - Store, at every trie node, max over the subtree's leaves of d_i. Static priors are precomputed; per-request d_i from a cheap scorer is one 3.5k-row GEMV plus a segmented max.
   - Bound(prefix) = prefix log-prob + subtree max. B&B stays exact.
   - Under trie normalization the bound stays tight, because single-child chains add 0. This turns M1 into "exact decoding for any additive item-level objective".
3. **Quantized search, FP16-certified output.**
   - Weight-only INT4/INT8 (Marlin) for forwards 1 and 2 is the dominant batch-1 lever, because the regime is weight-bandwidth-bound. Yet FlashRec saw only 70–76% beam overlap at FP8, and 2609.07901 shows INT4 concentrates choices.
   - Proposal: search with quantized logits, widen every bound by a measured per-level ε_q, then re-score the candidates within 2ε_q of the threshold in FP16 (only a few rows).
   - Result: an FP16-faithful top-K with a stated failure rate. This also fills ledger gap (f) with a ranking-faithfulness metric rather than only HR deltas.
4. **Dynamic, fan-out- and entropy-aware entry layer.**
   - Pick the entry layer per prefix (for example 18 / 22 / 24) from trie fan-out and the parent's next-level entropy. Both are available after forward 1.
   - In the tree-attention forward, rows join the residual stream at their own layer, so lower layers process fewer rows.
   - BRIDGE, IBA, LASAR and LoopRec show adaptive depth helps *trained* models. Post-hoc adaptive entry on a *frozen* LLM is open, and per-entry-layer bridges cost about 10M parameters each.
   - Pair it with idea 1 so that misrouted rows are caught near the boundary.
5. **Session-incremental prompt KV plus cached first-level rows (throughput regime).**
   - At B≥64 we are compute-bound (1,447 users/s on H100). There the ~100 prompt rows per user dominate forward-1 FLOPs.
   - A user's prompt grows by one item per request, so keep the prompt KV per session (MTServe, RelayGR, HBF and UniPinRec show this works at scale) and prefill only the appended item.
   - Optionally reuse the per-user first-code subtree when the history suffix is unchanged.
   - Not useful at batch 1, which is weight-bound. Potentially ~2× throughput at large batch, by FLOP counting.

**Also open, and cheap to add as experiments:**
- Use certified exact top-K as the teacher list for SLIDE-style one-pass students, or as RL rollout groups (Sibling-GRPO, V-STAR, PROMISE). No paper trains on exact-search outputs; 2611.01072 and DiGR compute exhaustive references offline only.
- Report the BB-style certification rate of *beam* outputs on MiniOneRec, to show how often beam-10/50 already equals the exact top-K.

---

## 4. What changes for positioning (concrete edits to `paper_draft.md`)

1. **§9 "Search and scoring".**
   - Remove "to our knowledge no prior work … gives a per-request optimality certificate".
   - Replace it with: BB (2609.33745) introduced a prefix-bound top-K certificate as *post-hoc completion after beam search* (full-vocab scores, ≤80 teacher-forced completions, certification 4–78% of queries, small T5).
   - We make the certificate the *primary* search: branch-and-bound from the root with tree-attention batch expansion, trie-normalized tight bounds, ~K expansions, 100% certified, 2.9 ms on a 1.5B LLM.
   - Also cite 2611.01072 (first-step losses; forward/cache overhead dominates beam wall time), which motivates scoring all first codes in one forward.
2. **Normalization claim.**
   - Replace "no one measures how normalization choices change quality" with: ZeroGR (2611.04715) adopts valid-child log-softmax for compute and reports negligible ranking change (99.85–99.90% overlap, ML-1M, small model).
   - Then add: PinGER, RTP-LLM local commerce and BB explicitly keep full-codebook or full-vocabulary log-probs.
   - Our contribution becomes: on an LLM recommender the choice matters (+1.2 pt HR@10), and it is what makes exact search cheap (single-child prefixes free, bounds tight).
   - Cite 2604.15739 for why trie normalization yields a proper item distribution while full-vocabulary normalization leaks mass.
3. **Late entry.**
   - The premise that deeper SID levels need less compute is now shared with BRIDGE (2611.07425), IBA (2607.12425), PFA (2610.39273), GR4AD LazyAR and SID-MLP.
   - Add DMTD (2510.11958), D³ (2503.08524), KV Prediction (2410.08391) and SwiftKV (2410.03960) next to SkipDecode and EAGLE.
   - Keep the novelty statement narrow: frozen backbone; the child's entry state is predicted from the parent's state at the same depth plus embeddings; composes with the certificate.
4. **Framing: "exact AR at the price of parallel decoders".**
   - 2026 work moves away from AR to cut latency: SPRINT 8–10×, OneChord, HGenPush and Netflix MTP, a single-level codebook (3→2 steps), ACT atomic tokens.
   - Netflix (2610.15864, ledger) shows AR beats MTP on quality.
   - Our numbers (2.9 ms certified top-10 vs Merge & Branch's 77 ms beam-100 and LatentRec's 590 ms beam-50 on 1.7B Qwen) argue that one need not give up AR coupling or exactness for latency. This is a stronger headline than "faster than beam".
5. **Value-aligned decoding (optional extension).** Kuaishou alone published UAGR, RGD, GR4AD-X and PROMISE in 2026, all patching premature pruning. A short section on certified decoding with additive item terms (idea 2) would connect the paper to the industrial agenda at little cost.

---

## References (primary pages consulted)

**Exact search and analyses**
- BB: https://arxiv.org/abs/2609.33745 (HTML v1)
- First-step coverage: https://arxiv.org/abs/2611.01072
- OneChord: https://arxiv.org/abs/2610.25805
- TCBSA: https://arxiv.org/abs/2610.32098
- V-STAR: https://arxiv.org/abs/2602.10699
- PROMISE: https://arxiv.org/abs/2601.04674
- DaV-Gen: https://arxiv.org/abs/2607.08365 and https://www.ijcai.org/proceedings/2026/363
- Disentangling GR: https://arxiv.org/abs/2610.08716
- Merge & Branch: https://arxiv.org/abs/2610.37973
- PATH: https://arxiv.org/abs/2608.29179

**Normalization and constraints**
- ZeroGR: https://arxiv.org/abs/2611.04715
- PinGER: https://arxiv.org/abs/2610.33435
- RTP-LLM local commerce: https://arxiv.org/abs/2611.06323
- AR-NTP ≡ FV-MLE: https://arxiv.org/abs/2604.15739
- PRISM: https://arxiv.org/abs/2601.16556
- UAGR: https://arxiv.org/abs/2611.02981
- RGD: https://arxiv.org/abs/2607.25344
- GR4AD-X: https://arxiv.org/abs/2610.38378
- Gryphon: https://arxiv.org/abs/2606.08604
- Gryphon-v2: https://arxiv.org/abs/2608.06213
- GPR: https://arxiv.org/abs/2511.10138
- vLLM RFC: https://github.com/vllm-project/vllm/issues/45815

**Depth and compute allocation**
- BRIDGE: https://arxiv.org/abs/2611.07425
- IBA: https://arxiv.org/abs/2607.12425
- LoopGR: https://arxiv.org/abs/2610.34157
- LoopRec: https://arxiv.org/abs/2610.30128
- SmartGR: https://arxiv.org/abs/2608.02048
- PFA: https://arxiv.org/abs/2610.39273
- Single-level codebook: https://arxiv.org/abs/2608.21012
- LESSRec: https://arxiv.org/abs/2610.37286
- LASAR: https://arxiv.org/abs/2605.10207
- LatentRec: https://arxiv.org/abs/2611.07866

**Systems**
- OneLA: https://arxiv.org/abs/2609.12399
- ReuseGR: https://arxiv.org/abs/2611.04259
- HBF: https://arxiv.org/abs/2609.07175
- UniPinRec: https://arxiv.org/abs/2606.00422
- GBLA: https://arxiv.org/abs/2606.07317
- GRP: https://arxiv.org/abs/2609.36688
- FLAME: https://arxiv.org/abs/2509.22681
- GEMs: https://arxiv.org/abs/2602.13631
- UniSGR-V2: https://arxiv.org/abs/2610.16132
- Rec2PM: https://arxiv.org/abs/2602.11605

**Parallel and lightweight decoding**
- SPRINT: https://arxiv.org/abs/2609.34306
- FLASH: https://arxiv.org/abs/2610.07402
- InforID: https://arxiv.org/abs/2608.09685
- HGenPush: https://arxiv.org/abs/2607.03362
- Netflix: https://arxiv.org/abs/2605.23312
- GenCDSR: https://arxiv.org/abs/2607.28659
- ACT: https://arxiv.org/abs/2611.00323
- Bit-factorized IDs: https://arxiv.org/abs/2610.17467
- SLIDE: https://arxiv.org/abs/2610.34764
- LIME-Rec: https://arxiv.org/abs/2608.01260
- CQ-SID: https://arxiv.org/abs/2605.14434
- IntRR: https://arxiv.org/abs/2602.20704
- STAMP: https://arxiv.org/abs/2604.05329
- PSG: https://arxiv.org/abs/2607.26427
- PushDualGen: https://arxiv.org/abs/2608.07989
- Multi-Decoder OneRec: https://arxiv.org/abs/2607.26500
- BARGE: https://arxiv.org/abs/2607.21028
- SPAR: https://arxiv.org/abs/2610.27898
- RoFiGR: https://arxiv.org/abs/2610.39470

**Training-side beam awareness**
- BEAR: https://arxiv.org/abs/2601.22925
- APAO: https://arxiv.org/abs/2603.02730
- GRC: https://arxiv.org/abs/2602.23639

**Quantization**
- 2609.07901: https://arxiv.org/abs/2609.07901
- LinkedIn: https://arxiv.org/abs/2502.14305

**General-LLM analogs**
- DMTD: https://arxiv.org/abs/2510.11958
- D³: https://arxiv.org/abs/2503.08524
- KV Prediction: https://arxiv.org/abs/2410.08391
- SwiftKV: https://arxiv.org/abs/2410.03960
- SPTS: https://arxiv.org/abs/2601.13155
- Speculative (2024): LASER https://arxiv.org/abs/2408.05676

**Unverified**
- CoSID: https://doi.org/10.1145/3773078.3841283 (ACM returned 403)
