# Inference optimization for SID-based generative recommendation: mechanism ledger, gaps, and positioning

Survey date: 2026-11-07. Scope: inference and serving optimizations for semantic-ID (SID) and LLM-based generative recommenders, 2023 to 2026.

**Method.** Primary sources: arXiv abstracts and HTML, GitHub READMEs and source (FlashRec, NVIDIA recsys-examples, xLLM via `gh`), and engineering blogs (NVIDIA, Spotify).

**Caveats.**
- Every speedup below is self-reported by the authors, on their own hardware and workloads. None were reproduced.
- Rows marked *(secondary)* come from summaries, not the primary text.
- Where I read code directly, the row says so.

---

## 0. Summary

- **The systems layer is crowded.** Every core mechanism in RapidTinyRec already has published or open-source precedent from 2025–2026:
  - prefill once, then share one physical prefix KV across beams
  - GPU trie or constraint tables
  - CUDA-graph decode
  - beam-KV reorder without copies
  - batching beams (and query heads) into rows against a shared context
  - restricted LM head
  - FP8

  The precedents are xGR/xLLM (JD), NVIDIA SID-GR, FlashRec (Sohu), GR4AD (Kuaishou), GRACE (Meta) and UniSGR-STARK (Alibaba). Claiming any one of these as new is not defensible.
- **What is missing is exactness and scoring semantics.** No published GR engine claims bit-exact parity with the HF reference at full beam width. Reported parity is approximate:
  - FlashRec: 86–92% beam-set overlap (BF16), 70–76% (FP8).
  - NVIDIA: top-1 match 1.000, mean top-K overlap 0.960.

  Engines also normalize scores differently and say nothing about it:
  - FlashRec: restricted per-codebook log-softmax (verified in source).
  - STATIC: full-vocabulary log-softmax, then mask.
  - xGR: mask added to logits before softmax.
  - NVIDIA: `ignore_eos` fixed-length output, which drops the suffix.

  No one measures how these choices change ranking quality.
- **Two "gaps" from the request are already taken.**
  - (d) prefill-only / distilled SID heads: SID-MLP, Deco, NEZHA, L2D, VaLiDRec, RPG.
  - (g) per-mechanism ablations: partially covered by GR4AD's QPS table.
- **Open gaps.**
  - (a) forced-suffix fusion and its ranking effect: open.
  - (b) normalization effect on ranking: open.
  - (c) exact exhaustive trie scoring with tree attention: open as an efficient method. DiGR did it by brute force (5 s per user).
  - (e) post-hoc early exit on SID steps: mostly open. GR4AD's LazyAR is the closest, and it is a trained architecture.
  - (f) weight-only INT8/INT4 effect on ranking: open. Only FP8 W8A8 has been studied.
- **Batch-1 reference points are sparse.** The best comparable number is NVIDIA SID-GR: 17.6 ms at batch 1 (Qwen3-1.7B, H100, BF16, beam 256, 1,000-token context, 3 output tokens). Nobody reports a 1–2B LLM recommender at beam 50 on a workstation GPU.

---

## 1. Mechanism ledger

The "Attacks" column uses this key:
- **P**: prefill
- **W**: decode weight reads
- **A**: attention / KV
- **B**: beam ops (top-K, sort, reorder)
- **C**: constraint check
- **H**: LM head / vocabulary
- **S**: scheduling / host overhead
- **Q**: precision
- **M**: model or architecture change

### 1A. GR serving engines and runtimes

| System (org, date, link) | Mechanisms | Attacks | Reported result (setting) | Quality / exactness | Code |
|---|---|---|---|---|---|
| **FlashRec** (Sohu MPTC; repo created 2026-09-03; github.com/sohu-mptc/FlashRec; no paper, `@software` cite) | Built on mini-sglang.<br>Wide-beam CUDA graphs, including beam expansion, captured at multiples of *n*.<br>Fused CUDA trie kernel (dense or CSR).<br>**Restricted per-codebook LM head with restricted `log_softmax`** (verified: `python/flashrec/logits.py` slices LM-head rows per codebook and runs `F.log_softmax` over only that slice).<br>Radix prefix KV cache with longest-prefix scheduling.<br>Cross-request decode waves under a beam-row budget.<br>FP8 W8A8 per-channel weights and FP8 KV cache, with fused norm/act→FP8 and QK-RoPE+KV-write. | W, B, C, H, S, Q, P | RTX 5090, OneRec-1.7B, RecIF-Bench, n=50 saturation: 28.1 QPS. That is 1.54× SGLang-0801, 2.02× SGLang-master, 2.21× TRT-LLM, 7.2× vLLM.<br>SoHuRec-1.7B: 220 QPS (n=50), ~48 QPS (n=512).<br>**No batch-1 latency.** | vs HF BF16: best SID matches at n=1–512; **beam-set overlap 86–92%**.<br>FP8 on a non-FP8-trained checkpoint: 70–76% overlap at n≥50.<br>Recall@32 0.034 for all engines.<br>README warns not to cite concurrency-1 recall. | Yes |
| **NVIDIA SID-GR inference** (recsys-examples `examples/sid-gr-inference`; blog 2026-08-20) | `ContextKV` holds the shared prompt once, `BeamKV` the short per-beam history, `BeamPath` the ancestry.<br>`gr_decode_atten` CuTe kernel reads the split layout directly.<br>Direct pool-view decode CUDA graph; piecewise prefill graph.<br>Item-constrained top-K with tries and masks.<br>Request × active-beam continuous batching.<br>Fixed, scheduled and **score-margin** dynamic beam policies.<br>Fused beam combine on Blackwell.<br>Fixed 3-token output with `ignore_eos` and special-token suppression. | P, A, B, C, S | H100, Qwen3-1.7B, BF16, beam 256, 3 output tokens.<br>**Batch 1: 17.6 ms at ctx 1000** (prefill 7.0 + decode 9.95) vs SGLang PR 33.5 ms. **42.3 ms at ctx 5000** vs 94.6 ms.<br>2.14–2.27× at batch 4–8.<br>Online: 1.85× req/s. | Offline top-1 exact match 1.000; mean top-K overlap with SGLang 0.960.<br>Log-softmax, top-K and beam selection run outside the graph (roadmap).<br>Quantization is listed as future work. | Yes |
| **xGR** (JD + Beihang et al.; arXiv 2512.11529; v1 2025-12-12, v2 2025-12-19, v3 2026-06-27) on **xLLM** (github.com/jd-opensource/xllm) | **xAttention**: shared prompt KV stored once plus token-granular unshared beam KV. Shared and unshared stages run separately and merge with online softmax; a decision-tree regressor picks the compute-unit split.<br>**Early sorting termination**: a size-K min-heap; each beam's sorted leaves are scanned until a log-prob falls below the heap top (exact for top-K).<br>**Mask-based item filtering** with the mask added to logits before softmax: dense at step 1, sparse in-place at the last step.<br>Buffer reuse; fixed-step scheduler; whole-graph multi-step execution with no per-step D2H.<br>Multi-stream overlap of host-side mask generation and H2D copies. | A, B, C, S | Ascend (64×16 NPUs) and H800.<br>Qwen3 0.6–4B and OneRec 0.1–3B; beam 128/256/512; 3 decode steps.<br>Under P99 ≤ 200 ms: **≥2.89× xLLM throughput (Qwen) and ≥8.05× (OneRec)** in v3. Earlier versions claimed ≥3.49×; deployment length is given as 3 months in one version and 6 in another.<br>Kernel: about 6.6× lower latency than PagedAttention. | Exactness of the optimizations is not stated. | xLLM is public. The xGR-specific release is unclear. |
| **Spotify + NVIDIA TensorRT-LLM** (Spotify Eng. blog, 2026-10-16) | Beam search moved inside the TRT-LLM sampler (v1.2.0rc3), with multi-sequence beam output and overlap scheduling.<br>Out-of-tree Qwen3 plugin with a **soft-token user embedding**: one projected token replaces textual history.<br>Multi-item scoring API (v1.3.0rc20). | S, B, P | Qwen3-1.7B: **up to 54% cost reduction** on beam workloads and 7% on scoring.<br>No latency, beam width or prompt length given in the text. | Not discussed. No constrained decoding is mentioned.<br>Quantization is "future". | Upstream in TRT-LLM releases |
| **SnapLGR** (Snap; arXiv 2607.28895, v3 2026-08-18) | TRT-LLM CUDA beam search; decentralized worker loop. | B, S | A100: **beam 32→96 cost 45% of throughput** (Recall@32 +11.56%). | Online A/B gains vs a TIGER-style baseline. | No |
| **GRACE** (Meta; arXiv 2608.00938, 2026-08-02/04) | Encoder–decoder ads retriever.<br>Beams become the *query-sequence* dimension of cross-attention, so one request K/V is shared.<br>Triton self-attention with block-diagonal coalescing of short beam rows.<br>Paged KV; beam reorder copies block IDs, not data.<br>**Whole SID loop in one CUDA graph** (forward, constraint, top-K, reorder).<br>Bitmask and Bloom-filter targeting constraints (GTM).<br>Dynamic beam (1, 512, 1024, 1024). | A, B, C, S | GH200, batch 16, beam 1024, 4-level SID, 3-layer decoder.<br>**Decoder P50 196.7 → 17.8 ms (11.1×)**; full model P50 51 ms.<br>Cross-attention 68× and self-attention 23–26× vs the best of FA2/FA3. | Cross-attention reshape is "layout-only". GTM can give false positives, removed by post-filtering. | No |
| **GR4AD** (Kuaishou ads; arXiv 2602.22732, v3 2026-04-02) | **LazyAR**: the first 6 of 9 decoder layers ignore the previous SID and run once per level for all beams; only the last 3 run per beam.<br>**Dynamic beam width** (e.g. 128-256-512 instead of 512³) and traffic-aware beam scaling.<br>Beam-shared KV.<br>**TopK pre-cut**: per-beam top-k, then a global top-k.<br>FP8.<br>Result cache with a ~1 min TTL. | M, A, B, Q, S | L20: **500+ QPS per GPU at <100 ms**.<br>Per-mechanism QPS: beam-shared KV +212.5%, TopK pre-cut +184.8%, FP8 +50.3%, cache +27.8%, LazyAR ~2× QPS, DBW 128-256-512 +45%. | DBW costs −0.15% revenue; FP8 about −0.1% revenue; LazyAR a "marginal" drop; TopK pre-cut is exact. | No |
| **OneRec V1 infra** (Kuaishou; arXiv 2506.13695) | TensorRT graph compilation; custom plugins (cross-attention, MoE); batching plus MPS. | S, A | L20: **5× throughput**, 28.8% inference MFU, OPEX 10.6% of the legacy cascade.<br>K=512 outputs.<br>No latency reported. | No exactness discussion. | Partial (OpenOneRec) |
| **OneRec-V2 FP8 PTQ** (Kuaishou; arXiv 2603.11486, 2026-03-12) | FP8 post-training quantization integrated into the serving stack. | Q, W | **−49% end-to-end latency, +92% throughput.** | "No degradation in core metrics" online. Offline ranking deltas are not given in the abstract. | No |
| **UniSGR / STARK** (Alibaba AIDC; arXiv 2607.04068, 2026-07-05) | Tree-attention beam decoding: candidates are concatenated along the sequence with an ancestor-only mask; shared prefixes are stored once. | A, B | **Batch 1: 14.1 ms avg / 15.6 ms P99 vs 30.9 ms without STARK**; 0.8B MoE.<br>Beams (512, 512, 1024).<br>Hardware unstated. | Claims identical beam results; no metric is reported. | No |
| **AliExpress ASGR** (Alibaba blog) *(secondary)* | Custom beam search replacing `torch.generate`; TRT-LLM; mixed-precision O4 (cast nodes 233→8); SDPA rewrite. | B, S | **−35% latency, +40% peak throughput** vs `torch.generate`. | n/a | No |
| **SGLang beam-search PRs** (#15645, #31626; cswuyg branches) | General beam search used as the baseline by NVIDIA and FlashRec. | B | Baseline only. | n/a | Yes |

### 1B. Constraint / trie implementations

| Work | Mechanism | Attacks | Result | Exactness / normalization | Code |
|---|---|---|---|---|---|
| **STATIC** (Google/YouTube; arXiv 2602.22647, KDD'26, v3 2026-10-07) | Trie flattened into a static CSR transition matrix; dense lookup for the first 2 levels; branch-free vectorized node-transition kernel (JAX). | C | TPU v6e, 3B Gemini-based model, beam 70, L=8, \|V\|=2048, ~20M constrained items: **+0.033 ms per step (0.25% of inference)**; 948× vs a CPU trie. | "Lossless". **Algorithm 1 applies LogSoftmax over the full vocabulary, then masks**, so scores stay full-vocabulary normalized (my reading of the algorithm; the text does not discuss it). | Yes |
| **FlashTrie** (arXiv 2607.10044, 2026-07-10; commercial search ads, affiliation not on abstract page) | Integer-aware succinct trie (Narrow-LOUDS / MARISA redesign) in HBM; one CUDA kernel for expansion, validation and pruning; parallel merge-sort prune. | C, B | 800M keywords, beam ≤1000: **<3 ms trie search, up to 24× vs multi-threaded CPU**; +0.71% revenue. | Exact. | "After review" |
| NEZHA hash-set verifier (below) | Integer-encode the SID and check it against a hash set after drafting. | C | Removes model calls from verification. | Invalid IDs are treated as draft failures. | Yes |

### 1C. Speculative and drafting methods

| Work | Mechanism | Attacks | Result | Quality | Code |
|---|---|---|---|---|---|
| **AtSpeed** (NUS/USTC; arXiv 2410.05165, ICLR'25) | N-to-K speculative beam decoding. A LLaMA-68M draft is aligned to the target's constrained top-K (AtSpeed-S) or relaxed acceptance is used (AtSpeed-R). **Tree attention** flattens drafted beams for verification. | W | LC-Rec LLaMA-7B, A5000.<br>**~1.8–2.0× strict, ~2.05–2.11× relaxed (max 2.51×)**; γ=4, draft beam 40, K≤20. | Strict mode is lossless; relaxed mode differs in the third decimal of Recall/NDCG. | Yes |
| **NEZHA** (Alibaba / CityU; arXiv 2511.18793, WWW'26) | Self-drafting RNN head inside the LLM, fed by L placeholder tokens in one prefill. Beam search runs on the head; hash-set verification replaces target-model verification. | W, P | Public sets, beam 10: **Qwen ~48–50 ms → 4–5 ms; Llama-1B ~74–79 → 7–8 ms** (hardware unstated).<br>Production: >1000 ms → <30 ms (includes queuing).<br>Taobao search ads since 2025-10. | Comparable to beam search; plain MTP is much worse. | Yes |
| **PAD-Rec** (USTC + Kuaishou; arXiv 2604.27747, 2026-04-30) | Position-aware drafter (item and step position embeddings, gated) for list-wise SID generation. | W | **Up to 3.1× wall-clock**; scaling tested 1B–8B. | "Largely preserving" quality. | Yes (per alphaXiv) |
| SpecGR (arXiv 2410.02939) | Speculation used for inductive (new-item) recall. | — | Not a speed method. | — | — |

### 1D. Decoder lightening, parallel or prefill-only SID generation

| Work | Mechanism | Attacks | Result | Quality | Code |
|---|---|---|---|---|---|
| **OneRec-V2 lazy decoder-only** (arXiv 2508.20900) | Drops the history encoder. The context is processed once, and lazy decoder layers cross-attend without K/V projections. | M, P | 1B: **18.9 vs 296.4 GFLOPs; −94% compute**. | Same loss (~3.27). | Partial |
| **GR4AD LazyAR** (above) | Layer-level beam sharing for decode steps 2 and later. | M, W | ~2× QPS. | Marginal drop. | No |
| **SID-MLP** (UCSD + Snap; arXiv 2605.12617, 2026-05-12) | Distills the AR decoder into per-position MLP heads over a once-computed user context. **Motivation: SID prediction difficulty drops sharply after the first token.** | M, W | **RTX A6000**, batch 32, beam 50: **8.74× (0.27 vs 2.36 ms/sample)**. Teacher is TIGER, 4.59M params. SID-MLP++ reaches 10.25×. | Matches teacher (non-inferiority within 1%). Appendix includes an LC-Rec 7B adaptation. | Yes |
| **Deco** (arXiv 2610.32980, v2 2026-10-26) | Backbone runs once to produce a "next-item concept"; a 2-layer KV-cached decoder runs the whole beam. | M, W | H100: **beam search is 91% of generation time at beam 100; search 14.9 → 2.1 ms**; 3.2–8.6× throughput at p95 ≤ 50 ms. With a 1.5B backbone it matches a 69M teacher's throughput. | Equal or better NDCG. | Not seen |
| **L2D** (NUS/USTC; arXiv 2509.11524, EMNLP'25) | Decode in latent space by matching the final hidden state to item representations. | W | **>10× faster** than language-space decoding. | Equal or better. | Not linked |
| **VaLiDRec** (arXiv 2607.25209, 2026-07-28) | Token-set prediction in one forward pass, aggregated over all item SIDs (no beam). | W, B | **87.49× vs LC-Rec.** | Beats baselines, per the abstract. | Not linked |
| **RPG** (Meta + UCSD; arXiv 2506.05781, KDD'25) | Long unordered SIDs (up to 64 tokens) predicted in parallel, with graph-guided decoding. | W, B | RTX 3090, ~13M params: **~15× faster than TIGER, ~25× less memory**. | +12.6% NDCG@10. | Yes |
| MaskGR (Snap/UT, arXiv 2511.23021); MDGR (Alibaba Intl., arXiv 2601.19501) | Masked-diffusion parallel SID denoising. | W | Efficiency claims only. | Comparable or better. | MaskGR yes |
| **DiGR** (arXiv 2602.07847 v2) | Direct item ranking.<br>**Also brute-force exhaustive AR scoring of LC-Rec** (Qwen3-4B) over every legal SID. | — | A100, Instruments (9,922 items): **exhaustive 5,052 ms/user vs beam-50 683 ms vs beam-100 981 ms**. | Exhaustive NDCG@10 0.0842 vs 0.0802 (beam 50) and 0.0816 (beam 100). Theorem: per-position marginals cannot preserve exact ranking. | — |
| **Netflix AR vs MTP** (arXiv 2610.15864, 2026-10-11) | Controlled comparison of constrained beam AR vs multi-token prediction. | W | AR throughput is 0.57× MTP (Qwen3-8B) and 0.18× (Qwen3-0.6B). | AR is better: +2.0% / +3.5% nMRR, +7–13% on discovery. Uses length-normalized log-prob. | No |
| PinRec (Pinterest; arXiv 2504.10507, v7 2026-06) | Multi-token generation; "efficient serving" section. | W | No numbers in the abstract. | — | No |

### 1E. Prefix / KV caching for user histories and prefill-only ranking

| Work | Mechanism | Result |
|---|---|---|
| **HSTU / M-FALCON** (Meta; arXiv 2402.17152, ICML'24) | Micro-batched candidate scoring with a modified mask; the user prefix is cached and amortized. HSTU attention kernel. | 1.5–2.99× (v2); serves 285× more complex models at DLRM cost; kernel 5.3–15.2× vs FA2 at 8k. |
| **MTGR** (Meituan; arXiv 2505.18654) | User-level compression (shared user tokens across candidates). | Inference cost −12% *(secondary)*. |
| **MTServe** (Meituan / WHU / NVIDIA; arXiv 2604.22881) | Hierarchical GPU + host-RAM cross-request KV cache for long histories; async transfer; locality-aware eviction. | Up to 3.1×, >98.5% hit rate. |
| **RelayGR** (Huawei; arXiv 2601.01712) | Pre-infers the user-prefix KV in an early pipeline stage and relays it to ranking via affinity routing (Ascend). | 1.5× longer sequences; 3.6× SLO-compliant throughput. |
| **Bat** (ZJU + Alibaba Taobao/Tmall; ASPLOS'26) | Bipartite attention: either the user or the item can serve as the cacheable prefix, with position and mask adjusted; hotness-aware scheduling. | Numbers not verified. |
| **RcLLM** (arXiv 2605.07443, ICDCS'26) | Non-prefix block KV reuse (replicated user caches, sharded item caches) with selective attention correction. | TTFT 1.31–9.51× lower than prefix caching. |
| **LONGER / OneTrans** (ByteDance; 2505.04421 / 2510.26104) | Cross-candidate and cross-request KV caching; pyramid token pruning. | LONGER: throughput loss −40% → −6.8% *(secondary)*. |
| **LinkedIn** (SGLang prefill-only, GRMIS, 360Brew distillation; MLSys'26 job-search SLM) *(secondary)* | Prefill-only scoring that reads only last-token logits; multi-item-scoring masks; GRMIS kernel 2× over SDPA; shared-prefix KV; 50% structured pruning. | 75× throughput claim is secondary and unverified. |
| **Netflix GenRec** (arXiv 2608.10257) | Prefill-only ranker on vLLM; context compaction ~5k → 1.7k tokens *(secondary)*. | ~3× cost reduction *(secondary)*. |
| HLEM (HKBU; arXiv 2605.04450) | Joint HBM split between embedding cache and KV cache, with reuse-aware routing. | — |
| **EARN** (NUS/USTC; arXiv 2507.00715, KDD'25) | Early layers compress the prompt into boundary register tokens; later layers attend only to those. | Up to 3.79× speedup, −80.8% KV. |
| **SLMRec** (ICLR'25, arXiv 2405.17890) | Layer-redundancy finding; distills an LLM recommender into fewer layers. | 13% of params; up to 8.0× inference. |
| OneRec-Think Think-Ahead (arXiv 2510.11639) | Reasoning precomputed offline; online completes only the tail. | +0.159% stay time. |

### 1F. General LLM primitives relevant to GR

| Primitive | Relevance |
|---|---|
| Hydragen (arXiv 2402.05099); FlashInfer cascade / shared-prefix (arXiv 2501.01005) | Shared-prefix attention: batches queries across beams against one prefix KV. Your "shared physical prefix KV" plus "fold GQA heads into rows" belongs to this family; GRACE's "beams as query dimension" is the GR instance. |
| SGLang compressed-FSM **jump-forward** decoding (arXiv 2312.07104); XGrammar (arXiv 2411.15100) | Emits multiple forced tokens in one forward. This is the closest prior for forced-suffix fusion (gap a). |
| SpecInfer / Medusa tree attention (arXiv 2305.09781, 2401.10774) | Tree masks for scoring many branches in one pass (gap c). |
| Grammar-Aligned Decoding (Park et al., NeurIPS'24, arXiv 2405.21047) | Shows that mask-then-renormalize constrained decoding distorts the model distribution. This is the closest general-LLM prior for gap (b). |
| Wu et al., "Constrained Auto-Regressive Decoding Constrains Generative Retrieval" (SIGIR'25, arXiv 2504.09935) | Theory: a KL lower bound from constraints and the sub-optimality of marginal-based beam search in GR. This is the closest GR theory for (b) and (c). |

### 1G. Beam-policy and early-termination methods

| Work | Kind | Exact? |
|---|---|---|
| xGR heap-based early sorting termination | Top-K selection shortcut | Yes |
| GR4AD TopK pre-cut (per-beam top-k, then global) | Top-K selection shortcut | Yes |
| NVIDIA score-margin beam policy | Beam shrinking | No; the README says real quality gates are still to do |
| GR4AD dynamic beam width / traffic-aware beam scaling | Beam schedule | No |
| GRACE dynamic beams (1, 512, 1024, 1024) | Beam schedule | No |
| Taobao CQ-SID: ~2× smaller beam via cluster SIDs *(secondary)* | Beam size reduction | No |
| GRID handbook (arXiv 2507.22224): unconstrained beam is much cheaper than constrained, with similar quality on small sets | Measurement | n/a |
| Spotify GLIDE (arXiv 2603.17540): beam 30 gives +27% Recall@30 vs greedy; 98% valid SIDs without constraints | Measurement | n/a |

### 1H. Quantization evidence for GR, complete as far as I found

| Source | Scheme | Finding |
|---|---|---|
| OneRec-V2 PTQ | FP8 | −49% latency; no online degradation |
| GR4AD | FP8 vs FP32 | +50.3% QPS; −0.1% revenue |
| FlashRec | FP8 W8A8 per-channel plus FP8 KV | 70–76% beam-set overlap on a non-FP8-trained checkpoint; 88–96% on FP8-trained SoHuRec. Overlap only, no recall delta |

- **No weight-only INT8/INT4 (AWQ, GPTQ, RTN) study of SID-GR ranking quality was found.**
- Spotify and NVIDIA both list quantization as future work.

---

## 2. Gaps: what is and isn't published

Closeness scale: **High** means the core idea is published in GR. **Medium** means it is published in adjacent work or partially. **Low** means only general-LLM analogues exist.

### (a) Fusing forced suffix tokens ("\n", EOS) into one forward, and whether scoring them matters

**Status: not published for GR. Closeness: Low–Medium.**

- **What engines do today.**
  - NVIDIA SID-GR fixes output length at 3 tokens with `ignore_eos` and suppresses special tokens, so it silently drops suffix scoring.
  - FlashRec decodes only SID codebook steps; its per-codebook head has no "\n" or EOS rows.
  - Nobody quantifies the ranking effect of dropping the suffix.
- **The mechanism exists in general LLMs.** SGLang's jump-forward decoding (arXiv 2312.07104) and XGrammar (arXiv 2411.15100) emit forced tokens in one forward pass. They are not applied to beam scoring, where the forced tokens' *log-probs* must still be accumulated.
- **Why the suffix matters, analytically.**
  - Under full-vocabulary normalization, log p("\n" | a,b,c) and log p(EOS | …, "\n") are item-dependent terms. They act like a learned per-item termination prior, so they can reorder the final 50.
  - Under trie-restricted normalization, a forced token has probability 1, so its term is exactly 0. Restricted normalization therefore makes the suffix free to skip.
  - So (a) and (b) are coupled.
- **Why fusion is exact.** Each final beam has exactly one admissible continuation, so the suffix steps only re-score the 50 beams and never prune. Teacher-forcing ["\n", EOS] as a 2-token-per-beam forward is exact at the level of math. Bit-exactness in FP32 depends on kernel shapes.
- **What it saves.** One full weight pass out of five at batch 1, or two passes if the suffix is dropped.

### (b) Restricted or per-codebook normalization vs full-vocabulary: effect on ranking quality

**Status: not measured anywhere I could find. Closeness: Medium (theory and implementations exist; no measurement).**

- **Implementations diverge.**
  - FlashRec: per-codebook `log_softmax` (verified in source).
  - STATIC: full-vocabulary `LogSoftmax`, then mask.
  - xGR: mask added to logits *before* softmax, which renormalizes over valid children (per the paper's description).
  - GRACE: has a dedicated 512-token SID vocabulary, so normalization is naturally restricted.
  - NVIDIA: logits buffer is `[batch, beam, vocab]`. I assume full-vocabulary normalization; not verified.
- **Reported parity metrics confound normalization with precision.** FlashRec's 86–92% overlap vs HF uses codebook-constrained search, not trie-constrained, and mixes BF16 with normalization effects.
- **Theory.** Wu et al. SIGIR'25 (arXiv 2504.09935) bounds the KL error from constraints. Grammar-Aligned Decoding (arXiv 2405.21047) shows mask-and-renormalize distorts LLM distributions. Neither measures Recall or NDCG for SID recommenders.
- **What a measurement would show.** Three scoring rules can produce different top-50 lists from the same logits:
  - full-vocabulary (the model's own distribution, matching MiniOneRec's training cross-entropy)
  - per-codebook restricted
  - per-trie-node restricted

  Their per-step difference is −log P(admissible set | prefix). That term is prefix-dependent, so it reorders across prefixes.
- **Why it is cheap to defend.** It is purely an inference-time choice, it is measurable on public checkpoints, and it bears directly on whether "restricted LM head" speedups are free.

### (c) Exact (non-beam) exhaustive trie scoring with tree attention for small catalogs

**Status: not published as an efficient method. Closeness: Medium.**

Closest prior work:
- **DiGR** (arXiv 2602.07847) scored every legal SID of LC-Rec (Qwen3-4B) **by brute force: 5.05 s per user on an A100** for 9,922 items. Exhaustive NDCG@10 was 0.0842 vs 0.0802 at beam 50, which is direct evidence that beam-50 loses quality.
- **UniSGR STARK** (arXiv 2607.04068) uses tree attention only over beam candidates.
- **AtSpeed** uses tree attention only to verify drafted beams.
- **ISD** (arXiv 2607.24995) calls exhaustive scoring "too expensive" and uses rank fusion instead.
- VaLiDRec, TopoGR and RPG score all items with *parallel per-position marginals*. DiGR's Theorem 1 proves that this cannot preserve the exact AR ranking.

Cost reasoning (my estimate; please check against your catalog):
- Exact AR scoring of all 3,459 items needs one token row per **internal trie node**: about |unique a| + |unique (a,b)|, likely around 3k rows. That is one causal, tree-masked forward over the ~100-token prompt KV.
- Under full-vocabulary normalization with the suffix, add about 2 × 3,459 leaf rows.
- That forward is compute-bound, while beam-50 at batch 1 is bandwidth-bound across about 5 passes.
- On an A6000 (~150 dense BF16 TFLOPs, 768 GB/s), 3k rows × ~2.6 GFLOP/row ≈ 8 TFLOP. That is tens of ms, the same order as the beam. On H100-class GPUs the ratio is similar.
- The defensible claim is therefore "exact top-K, no beam error, at latency comparable to beam-50 for catalogs up to roughly N", not "faster than beam".

### (d) Distilled lightweight SID decoder head, so the LLM runs once

**Status: published, repeatedly in 2026. Closeness: High. Not a gap.**

- **SID-MLP** (arXiv 2605.12617): per-position MLP heads, 8.74×, also measured on an RTX A6000; includes an LC-Rec 7B appendix.
- **Deco** (arXiv 2610.32980): concept vector plus a 2-layer decoder; includes a 1.5B-backbone variant.
- **NEZHA** (arXiv 2511.18793): draft head inside the LLM, with hash verification; deployed.
- **L2D** (arXiv 2509.11524), **VaLiDRec** (arXiv 2607.25209), **RPG** (arXiv 2506.05781), **OneRec-V2 lazy decoder**, **GR4AD LazyAR**.

What remains narrow:
- A head distilled onto a frozen *public LLM-rec checkpoint* (MiniOneRec), measured by *agreement with the LLM's own exact beam ranking* rather than only by Recall.
- Even that is close to SID-MLP's LC-Rec appendix and NEZHA. I would not lead with it.

### (e) Early exit or layer skipping for SID decode steps

**Status: mostly open. Closeness: Medium.**

- **GR4AD LazyAR** (arXiv 2602.22732) is the closest: 6 of 9 decoder layers are shared across beams for steps 2 and later, giving ~2× QPS. It is a *trained architecture* in a 0.16B purpose-built model, not post-hoc exit on a pretrained LLM.
- **SID-MLP**'s finding that difficulty drops after the first token is the direct motivation for skipping depth on later SID steps.
- **SLMRec** (arXiv 2405.17890) shows intermediate layers of LLM recommenders are redundant; it uses static distillation. **EARN** (arXiv 2507.00715) compresses the prompt after early layers.
- **No confidence-based or per-SID-level early exit on an LLM-based SID recommender was found.**
- General-LLM evidence (River-LLM arXiv 2604.18396; "Diminishing Returns of Early-Exit" arXiv 2603.23701) warns that wall-clock gains are often marginal because of KV-cache consistency. With only 2–4 decode steps and shared prefix KV, GR may be a better fit. That would be a testable claim.

### (f) Weight-only INT8/INT4 quantization effect on GR ranking quality

**Status: open. Closeness: Medium-low.**

- Only FP8 W8A8 has been studied:
  - OneRec-V2: online A/B, no degradation.
  - GR4AD: −0.1% revenue.
  - FlashRec: beam-overlap drop to 70–76% with on-load FP8 of a non-FP8-trained checkpoint. This is the most informative public datum, and it suggests SID beams are *more* sensitive than typical LLM benchmarks.
- No AWQ, GPTQ or RTN weight-only study with Recall, NDCG or beam-overlap for SID-GR exists that I could find.
- At batch 1, weight-only quantization is the lever that directly cuts the dominant cost (weight bytes per pass), so this gap is relevant to your setting.

### (g) An end-to-end "speedrun" ablation ladder from HF baseline to optimized, for GR

**Status: not published in this form. Closeness: Medium.**

- **GR4AD** Table 1/3 gives per-mechanism QPS increments. It covers throughput only, uses an internal 0.16B model and is not reproducible.
- **FlashRec, NVIDIA and xGR** report engine-vs-engine numbers.
- **AliExpress** reports a single "custom beam vs `torch.generate`" step (−35%).
- **Deco** reports a time breakdown (beam is 91% of time).
- None gives a *cumulative, reproducible, batch-1 ladder from HF `generate` with a correctness gate (bit-exact top-K) at each rung*.

---

## 3. Batch-1 (or near) latency reference points

| System | Model | HW / precision | Beam | Prompt | Output steps | Batch | Latency |
|---|---|---|---|---|---|---|---|
| NVIDIA SID-GR | Qwen3-1.7B | H100 / BF16 | 256 | 1,000 | 3 | **1** | **17.6 ms** (7.0 prefill + 9.95 decode); SGLang 33.5 ms |
| NVIDIA SID-GR | Qwen3-1.7B | H100 / BF16 | 256 | 5,000 | 3 | **1** | **42.3 ms**; SGLang 94.6 ms |
| UniSGR STARK | ~0.8B MoE | unstated | 512/512/1024 | unstated | 3 | **1** | **14.1 ms avg / 15.6 ms P99** (30.9 ms without tree attention) |
| NEZHA (public sets) | Qwen3 ~0.6–0.8B / Llama-1B | unstated | 10 | short | SID | unclear | Beam search **48–50 ms** (Qwen) / **74–79 ms** (Llama); NEZHA 4–8 ms |
| DiGR (LC-Rec) | Qwen3-4B | A100 | 50 / 100 | short | SID | per user | **683 ms / 981 ms**; exhaustive 5,052 ms (unoptimized) |
| GRACE | Encoder–decoder, 3-layer decoder | GH200 | 1024 | 128 context tokens | 4 | 16 | Decoder P50 17.8 ms; full model 51 ms |
| GR4AD | 0.16B | L20 / FP8 | ≤512 (dynamic) | long | 3 | serving | <100 ms at 500+ QPS |
| SID-MLP | TIGER 4.59M | **RTX A6000** | 50 | short | 4 | 32 | 2.36 → 0.27 ms/sample |
| FlashRec | 1.7B | RTX 5090 / FP8 | 50–1000 | — | 3–5 | — | **Throughput only, no batch-1 latency** |
| **RapidTinyRec (yours)** | Qwen2-1.5B | **RTX A6000**, FP32 exact / BF16 | 50 | 76–121 | 5 scored | **1** | **115 ms (FP32 bit-exact); 56.5 ms (BF16)**; HF 869 ms |

**Positioning.** This is my back-of-envelope estimate, not from any paper.

- **Weight bytes per pass.** Qwen2-1.5B has about 1.31B non-embedding parameters (2.62 GB in BF16). The tied 152k × 1536 LM head adds 0.47 GB per pass under full-vocabulary scoring. That is about 3.1 GB per pass.
- **Bandwidth floor.** At 768 GB/s, each pass costs at least 4.0 ms. Your scheme does 5 weight passes (prefill, then b, c, "\n", EOS), so the BF16 floor is about 20–25 ms. FP32 doubles that to about 40–50 ms.
- **Distance from the floor.** At 56.5 ms BF16 you are about 2.3–2.8× above this floor.
- **Levers ranked by bytes saved.** At batch 1 these, not further kernel work, are the remaining levers:
  - fuse or skip the forced suffix: −1 to −2 passes, 20–40%
  - restricted head on the b and c steps: ~15% per pass, but this changes semantics (gap b)
  - weight-only INT8 or INT4: −50% or −75% of bytes (gap f)
- **Comparing with NVIDIA.** Scaling NVIDIA's 17.6 ms (H100, 3.35 TB/s) by bandwidth to an A6000 gives about 77 ms. Their workload is heavier (prompt 10×, beam 5×) and has fewer steps (3 vs 5). So the honest statement is "same order of magnitude on a 4.4× lower-bandwidth GPU, with exact HF parity", not "faster than NVIDIA".

---

## 4. The five most defensible novelty claims

1. **An exactness-gated optimization ladder for LLM-based SID recommendation.**
   - What: the first published cumulative, reproducible ablation from HF `generate` (869 ms) to an optimized batch-1 engine (115 ms), with **FP32 bit-exact top-50 parity checked at every rung**. Precision (BF16, 56.5 ms) is reported as a separate rung with ranking-metric deltas.
   - Prior art reports engine-vs-engine throughput with *approximate* parity: FlashRec 86–92% overlap, NVIDIA top-1 match with 0.96 top-K overlap, GR4AD per-mechanism QPS on an internal model.
   - Defensibility: high. It is method and measurement, not a mechanism claim.
2. **A study of scoring semantics: normalization and forced-suffix accounting.**
   - What: quantify how full-vocabulary, per-codebook and per-trie-node normalization, combined with including or dropping the "\n"/EOS terms, change the top-K list and Recall/NDCG on MiniOneRec checkpoints.
   - Analytic point: forced tokens contribute exactly 0 under restricted normalization but item-dependent terms under full-vocabulary normalization.
   - Production engines already differ on these choices (FlashRec restricted, STATIC full-then-mask, xGR mask-before-softmax, NVIDIA `ignore_eos`) and none report the effect.
   - Defensibility: high. Nothing measured exists.
3. **Exact forced-suffix fusion (jump-forward for beam scoring).**
   - What: score the deterministic suffix of all beams in one teacher-forced 2-token forward. It is exact because each beam has a single admissible continuation, so no pruning happens. It removes one of five weight passes at batch 1.
   - Prior art: SGLang and XGrammar jump-forward, which do not accumulate log-probs over beams; NVIDIA and FlashRec drop the suffix instead.
   - Defensibility: medium-high. The mechanism is small, so pair it with claim 2.
4. **Exact exhaustive top-K for small catalogs with one tree-attention forward over the SID trie.**
   - What: compute the exact AR score of every item, removing beam-search error, at latency comparable to beam-50.
   - Prior art: DiGR shows exhaustive beats beam-50 (NDCG@10 0.0842 vs 0.0802) but needed 5 s per user by brute force; STARK and AtSpeed use tree attention only over beam or draft candidates.
   - Defensibility: medium. It holds only if you show the latency crossover; frame it as "exact at comparable cost up to catalog size N".
5. **A batch-1 roofline characterization plus the first weight-only quantization study for SID-GR ranking.**
   - What: show that LLM-based GR at batch 1 on a workstation GPU is weight-read-bound, rank the levers by bytes per pass times number of passes, and measure weight-only INT8/INT4 (RTN, AWQ or GPTQ) effects on beam overlap and Recall/NDCG against your FP32-exact reference.
   - Prior art is FP8 W8A8 only: OneRec-V2 online, GR4AD revenue, FlashRec overlap. An optional extension is per-SID-level early exit, where GR4AD LazyAR is the closest prior and is a trained architecture.
   - Defensibility: medium-high.

**Claims to avoid.** These are already published:
- shared prefix KV across beams (xGR, NVIDIA, GR4AD, GRACE, STARK)
- GPU trie or constraint tables (STATIC, FlashTrie, FlashRec, xGR)
- CUDA-graph decode (FlashRec, NVIDIA, GRACE, xLLM)
- KV reorder without copies (GRACE block-ID copy, NVIDIA BeamPath, xGR token-granular unshared KV)
- head or beam folding against a shared prefix (Hydragen, GRACE)
- restricted LM head (FlashRec)
- distilled lightweight decoders (SID-MLP, Deco, NEZHA)
- speculative beam decoding (AtSpeed, NEZHA, PAD-Rec)

Cite them as engineering lineage, not contributions.

---

## References (primary sources consulted)

**Serving engines and runtimes**
- FlashRec: https://github.com/sohu-mptc/FlashRec (README; `python/flashrec/logits.py`)
- NVIDIA SID-GR: https://github.com/NVIDIA/recsys-examples (`examples/sid-gr-inference/README.md`); blog https://developer.nvidia.com/blog/how-generative-recommenders-are-redefining-recsys-at-scale/ (2026-08-20)
- xGR: https://arxiv.org/abs/2512.11529 (v1 2025-12-12, v3 2026-06-27). xLLM: https://github.com/jd-opensource/xllm (`docs/.../generative_recommendation_design.md`)
- Spotify + NVIDIA TRT-LLM: https://engineering.atspotify.com/2026/10/advancing-generative-recommendation-inference-our-collaboration-with-nvidia-on-tensorrt-llm (2026-10-16)
- SnapLGR: https://arxiv.org/abs/2607.28895
- GRACE: https://arxiv.org/abs/2608.00938
- GR4AD: https://arxiv.org/abs/2602.22732
- OneRec: https://arxiv.org/abs/2506.13695
- OneRec-V2: https://arxiv.org/abs/2508.20900
- OneRec-V2 FP8: https://arxiv.org/abs/2603.11486
- OneRec-Think: https://arxiv.org/abs/2510.11639
- UniSGR: https://arxiv.org/abs/2607.04068
- AliExpress ASGR *(secondary)*: https://www.besthub.dev/articles/engineering-exploration-of-generative-recall-leveraging-large-models-for-next-gen-recommendation-retrieval-a69467166629

**Constraint / trie**
- STATIC: https://arxiv.org/abs/2602.22647; code https://github.com/youtube/static-constraint-decoding
- FlashTrie: https://arxiv.org/abs/2607.10044

**Speculative and drafting**
- AtSpeed: https://arxiv.org/abs/2410.05165
- NEZHA: https://arxiv.org/abs/2511.18793
- PAD-Rec: https://arxiv.org/abs/2604.27747
- SpecGR: https://arxiv.org/abs/2410.02939

**Decoder lightening and parallel generation**
- SID-MLP: https://arxiv.org/abs/2605.12617
- Deco: https://arxiv.org/abs/2610.32980
- L2D: https://arxiv.org/abs/2509.11524
- VaLiDRec: https://arxiv.org/abs/2607.25209
- RPG: https://arxiv.org/abs/2506.05781
- MaskGR: https://arxiv.org/abs/2511.23021
- MDGR: https://arxiv.org/abs/2601.19501
- DiGR: https://arxiv.org/abs/2602.07847
- Netflix AR vs MTP: https://arxiv.org/abs/2610.15864
- PinRec: https://arxiv.org/abs/2504.10507

**Prefix / KV caching and model compression**
- HSTU: https://arxiv.org/abs/2402.17152
- MTGR: https://arxiv.org/abs/2505.18654
- MTServe: https://arxiv.org/abs/2604.22881
- RelayGR: https://arxiv.org/abs/2601.01712
- RcLLM: https://arxiv.org/abs/2605.07443
- LONGER: https://arxiv.org/abs/2505.04421
- OneTrans: https://arxiv.org/abs/2510.26104
- Netflix GenRec: https://arxiv.org/abs/2608.10257
- EARN: https://arxiv.org/abs/2507.00715
- SLMRec: https://arxiv.org/abs/2405.17890

**Measurements and theory**
- GRID handbook: https://arxiv.org/abs/2507.22224
- Spotify GLIDE: https://arxiv.org/abs/2603.17540
- ISD: https://arxiv.org/abs/2607.24995
- Wu et al. SIGIR'25: https://arxiv.org/abs/2504.09935
- MiniOneRec: https://arxiv.org/abs/2510.24431

**General LLM primitives**
- Hydragen: https://arxiv.org/abs/2402.05099
- FlashInfer: https://arxiv.org/abs/2501.01005
- SGLang: https://arxiv.org/abs/2312.07104
- XGrammar: https://arxiv.org/abs/2411.15100
- Grammar-Aligned Decoding: https://arxiv.org/abs/2405.21047
- SpecInfer: https://arxiv.org/abs/2305.09781
- Medusa: https://arxiv.org/abs/2401.10774

**Not verified from primary text:** Bat (ASPLOS'26) numbers; LinkedIn MLSys'26 75× claim; MTGR −12%; LONGER −6.8%; CQ-SID beam halving; Netflix GenRec compaction numbers.
