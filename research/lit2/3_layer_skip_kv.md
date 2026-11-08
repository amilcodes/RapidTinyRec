# Layer skipping, late entry, early exit, KV reuse and position-independent caching: general LLM + recsys

Survey date: 2026-11-08. Scope: priors for RapidTinyRec's **late-entry bridge**, for **idea A** (history-side item-local lower layers and/or early-exit KV, turning the decoder-only model into an encoder + decoder by distillation), for **idea B** (deeper or native late entry via light fine-tuning), and for **(C)** layer-wise "where is the item decided" analysis.

**Method.** WebSearch, then each paper checked on its arXiv abstract page. For the closest priors I also read the arXiv HTML or the PDF text, extracted locally:
- Read in full text: FIRP, SwiftKV, EARN, RcLLM, Bat, GR4AD, OneRec-V2, OneTrans, HatLLM, the COMB "encoder for PIC" paper, AdaDecode and SkipDecode.
- Read in part: HSTU (M-FALCON section only).

**Caveats.**
- All numbers are self-reported by the authors. None were reproduced.
- A † after a venue means I recall it but did not see it on the page. Everything else (venue and date) was confirmed on the arXiv page or the proceedings page.
- Rows that were verified earlier in `research/lit_ledger.md` are marked *(ledger)*.

**Closeness key.** Each paper gets three ratings: **LE** (our late-entry bridge), **A** (idea A) and **B** (idea B). Each is rated **none / adj** (adjacent: same family, different mechanism or target) **/ DIRECT** (the same mechanism, or the same goal in the same setting).

---

## 0. TL;DR

1. **Late entry has a direct conceptual prior in recsys: GR4AD's LazyAR, "Late-Inject Autoregression"** (Kuaishou, arXiv 2602.22732).
   - **What it does:** in a 9-layer decoder, the first K=6 layers of each SID level run without the previous SID token. The previous token is fused in at layer K through a gated projection. The shallow trunk runs once and is shared across all beams and levels.
   - **Results:** +117% QPS, with revenue at +4.28% versus +4.32% without LazyAR.
   - **How it differs from ours:** it is a *trained architecture*, built with an auxiliary MTP loss that forces the shallow trunk to be predictive. It does **not** predict the child's residual stream from the parent's state.
   - **The closest *mechanistic* priors:**
     - **FIRP / Hidden Transfer** (Wu et al., NLPCC'24 / arXiv 2404.12022): a linear map from intermediate-layer states to "pseudo" layer-*t* states of future positions. These are injected mid-pass and refined by the remaining layers with attention to context, trained with KL.
     - **EAGLE** (feature + next-token embedding → next feature).
     - **SkipDecode** (later tokens skip lower and middle layers).
   - **None of them** does all of the following:
     - conditions on the known child token
     - is used without verification
     - is post-hoc on a frozen pretrained LLM
     - exploits the SID tree, so that no KV hole ever needs filling
2. **Idea A is partly done in recsys, but not as "distill a frozen GR LLM into item-local lower layers plus early-exit history KV."**
   - **Already done:**
     - Per-item, context-free KV for *candidate* items with block-diagonal masks: Bat (ASPLOS'26, also on Qwen2-1.5B) and RcLLM (ICDCS'26).
     - History tokens dropped above layer k: EARN (KDD'25), which needs full fine-tuning plus register tokens.
     - Upper-layer history queries pruned: OneTrans pyramid.
     - Context that is only K/V-projected, never encoded: OneRec-V2 lazy decoder, trained from scratch.
     - Shallow layers restricted to intra-item attention for *quality*: HatLLM (LoRA, no efficiency claim).
   - **Still open:**
     - (a) item-local lower layers for *history SIDs*, with an exact RoPE re-positioned per-item cache, recovered by *distillation* on a frozen GR model;
     - (b) SwiftKV-style early-exit KV for history tokens in GR;
     - (c) combining (a) and (b) so that the whole history KV is a per-item lookup and prefill shrinks to the last few template tokens.
3. **Biggest batch-1 opportunity: depth-staggered single-pass trie scoring** (experiment E1 in §4).
   - **The observation:** every bridge input (prompt-last state at layer *e*, an *a* node's layer-*e* state, a *b* node's layer-*e* state) exists *without running any upper layer*. So SID nodes can be injected at layer *e* while the same pass is still running.
   - **Prior:** FIRP injects pseudo-states mid-pass, but for drafting unknown tokens. No paper found does this for exhaustive tree scoring of known tokens with no added approximation.

---

## 1. Lower-layer skipping, late entry and cross-position state prediction

| Paper (venue, date) | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **GR4AD LazyAR** "Late-Inject Autoregression" (Kuaishou; arXiv 2602.22732, v3 2026-04-02; under review) | For SID level *t*, the first K of L layers see only the position embedding and the context, not s_{t−1}.<br>s_{t−1} is injected at layer K by gated fusion: layer-K state ⊙ proj(s_{t−1}), concatenated with s_{t−1}, then projected.<br>Production L=9, K=6.<br>The first K layers run once for all levels and are reused across beams.<br>An auxiliary MTP loss bypasses the fusion. | **+117% QPS** ("nearly doubles").<br>Revenue +4.28% vs +4.32% without LazyAR.<br>The authors say the gain is rec-specific. | Yes. Trained architecture (0.16B, 9 layers). | **DIRECT** (concept) / none / **DIRECT** (a native, trained late entry) |
| **FIRP** (Wu, Liu et al., PKU+Meituan; arXiv 2410.20488, NLPCC'24) and **Hidden Transfer** (same group; arXiv 2404.12022, 2024-04) | Per-step linear maps take layer-*t* states of the context and produce pseudo layer-*t* states for positions j+i (7B: t=15 then 20; 13B: 25/30/35).<br>The pseudo-states are concatenated mid-pass and run through the remaining layers with a custom mask (attend to context up to the target plus earlier pseudo-states).<br>Tree verification keeps the output lossless.<br>Layer ablation: accuracy is flat from layer 5 to 15, then drops. | 1.9–3× (e.g., GSM8K 2.6–2.8×, XSum ≈2×) on LLaMA-2 / Vicuna 7B and 13B. | Yes. K linear maps, KL to the shifted original distributions, 2 epochs on ShareGPT. | **DIRECT** (mechanism; but drafting, unknown tokens, verified) / none / adj |
| **SkipDecode** (Del Corro et al., MSR; arXiv 2307.02628, 2023-07) | Token-level exit layer decays monotonically with position.<br>Tokens *bypass lower and middle layers* (1 "warm-up" bottom layer, then jump to the top y layers), so later tokens attend to the top-layer KV of all earlier tokens.<br>The prompt runs full depth.<br>No KV holes, because exits are monotone. | 2–5× on OPT-1.3B/6.7B.<br>Flat up to 2×, then steady degradation (E2E, Reddit-TLDR, CNN-DM). | Yes. Fine-tunes the whole model with the skip schedule. | **DIRECT** (skip lower layers for later tokens; but the entry state is the token's own 1-layer state, not parent-predicted) / none / adj |
| **EAGLE** (Li et al.; arXiv 2401.15077, ICML'24†) | Autoregression at the second-to-top feature level. Input = feature sequence plus the token sequence advanced one step, i.e. *parent feature + child token embedding → child feature*. Drafts, then verifies. | 2.7–3.5× latency on LLaMA2-Chat 70B; lossless. | Yes (draft head; frozen target). | **DIRECT** (bridge input signature; top layer, draft) / none / adj |
| **EAGLE-2** (arXiv 2406.16858, EMNLP'24†) | Dynamic draft tree driven by calibrated draft confidence. | 3.05–4.26×, lossless. | Reuses the EAGLE head. | adj / none / none |
| **EAGLE-3** (arXiv 2503.01840, NeurIPS'25†) | Fuses low, mid and high features; predicts tokens; "training-time test" (trains on its own drafted states). | Up to 6.5×; 1.38× SGLang throughput at bs 64. | Yes. | adj (training-time test is a good recipe for multi-hop bridges) / none / adj |
| **Medusa** (arXiv 2401.10774, ICML'24†) | Extra heads on the last hidden state propose future tokens; tree attention verifies. | Medusa-1 >2.2×; Medusa-2 2.3–3.6×. | Heads only (Medusa-1, frozen backbone). | adj / none / none |
| **Hydra** (arXiv 2402.05109, 2024-02) | Medusa heads made sequentially dependent on earlier drafted tokens. | Up to 2.70× vs AR; 1.31× vs Medusa. | Heads. | adj / none / none |
| **Kangaroo** (arXiv 2404.18911, 2024-04) | Shallow sub-network plus a trained adapter acts as the draft; double early exit. | 1.68× on Spec-Bench; 88.7% fewer extra parameters than Medusa-1. | Adapter only. | adj (adapter bridging a depth gap) / none / adj |
| **Draft & Verify** (arXiv 2309.08168, ACL'24) | Self-speculative: draft by skipping intermediate layers, verify with the full model. | Up to 1.99× on LLaMA-2; lossless. | No. | adj / none / none |
| **SWIFT** (arXiv 2410.06916, ICLR'25) | On-the-fly adaptive choice of which intermediate layers to skip for self-speculation. | 1.3–1.6×; lossless. | No. | adj / none / none |
| **CLaSp** (arXiv 2505.24196, ACL'25) | A DP picks the skip set, using the last verification's hidden states as target; re-chosen after each verification. | 1.3–1.7× on LLaMA3; lossless. | No. | adj / none / none |
| **LayerSkip** (Meta; arXiv 2404.16710, ACL'24) | Layer dropout rising with depth plus an early-exit loss with a shared LM head. Self-speculative decoding reuses the draft's KV. | Up to 2.16× (CNN/DM), 1.82× (code). | Yes (pretrain, CPT or FT recipe). | adj / none / **DIRECT** (recipe to make a model robust to entering or exiting at other depths) |
| **CALM** (Schuster et al.; arXiv 2207.07061, NeurIPS'22 oral) | Per-token confidence-based early exit with calibrated, risk-controlled thresholds. Skipped-layer KV of exited tokens is filled from the exit-layer state ("state propagation"; from the paper body, not the abstract). | Up to 3× (T5). | Yes. | adj (the inverse of LE: exit, then copy state upward) / adj / none |
| **FREE** (arXiv 2310.05424, EMNLP'23) | Shallow-deep module; synchronized parallel decoding computes the deep layers of exited tokens alongside the next token; Beta-mixture threshold. | Faster than CALM, more robust (no numbers on abstract). | Yes. | adj / none / adj |
| **AdaDecode** (arXiv 2506.03700, ICML'25) | Intermediate LM heads (KL to final). A confident token is emitted early. Its remaining layers are *deferred* and batched with later tokens, which fills its deep KV. Rejection rule keeps parity. | Up to 1.73× (CodeLlama-34B); Llama3.1-8B 1.14–1.51×; consistency 0.996. | Heads only (~16M each). | adj / none / none |
| **LITE** (Varshney et al.; arXiv 2310.18581, 2023-10) | Instruction tuning with intermediate-layer losses enables confidence-based token exit. | 37.9% (7B) / 46.4% (13B) compute saved. | Yes (IT). | adj / none / adj |
| **D³ Position-Aware Depth Decay** (arXiv 2503.08524, v2 2025-11) | Training-free: token *i* runs floor(L·α^i) layers, so later tokens get less depth. | ~1.5× on Llama 7–70B; <1% drop on GSM8K/BBH. | No. | adj ("later tokens need less depth", as for SIDs) / none / none |
| **Unified Layer Skipping** (arXiv 2404.06954, 2024-04) | Input-independent, balanced skipping of intermediate layers at a target speedup ratio; batch- and KV-friendly. | Beats dynamic methods (MT, summarization). | Not stated. | adj / none / none |
| **Mixture-of-Depths** (DeepMind; arXiv 2404.02258, 2024-04) | Top-k router per layer picks which tokens get attention+MLP; static graph. | Iso-FLOP parity; up to 50% faster sampling. | Yes (pretrain). | adj / adj / **DIRECT**-ish (a native per-token depth, but from scratch) |
| **D-LLM** (Huawei; NeurIPS'24) | Per-layer, per-token execute/skip decision module, plus an eviction policy for the missing KV of skipped layers. | Lower compute and KV; no numbers in abstract. | Yes (FT). | adj / none / adj |
| **FlexiDepth** (arXiv 2503.23798, COLM'25) | Plug-in router plus adapter on a *frozen* LLM decides per-token layer skips. | Llama-3-8B skips 8 of 32 layers at full benchmark parity. **No wall-clock speedup.** | Router and adapter only. | adj (frozen-model adapter replacing skipped layers) / none / adj |
| **Router-Tuning / MindSkip** (arXiv 2410.13184, EMNLP'25) | Train only a router on a small dataset; "attention with dynamic depths". | Up to 21% speedup at −0.2%. | Router only. | adj / none / adj |
| **Learning to Skip the Middle Layers** (arXiv 2506.21103, 2025-06) | Gate skips a symmetric middle span; gated attention so later tokens ignore skipped positions. | **Negative:** no gain over smaller dense baselines at the scales tested. | Yes. | adj (negative result worth citing) / none / adj |
| **Dynamic layer selection in decoder-only transformers** (arXiv 2410.20022, 2024-10) | Study: pretrained decoders are much more robust to *layer skipping* than to *early exit*. Per-token control from hidden states is hard; per-sequence control is promising (oracle uses 23.3% of layers). | Analysis. | No. | adj / none / adj |
| **ADEPT** (arXiv 2601.03700, 2026-01) | Token-level early exit in prefill *and* generation. Decouples sequential dependence to generate skipped-layer KV. | Up to 25% efficiency gain in generation. | Not stated. | adj / adj / none |
| **Tandem Transformers** (Google; arXiv 2402.08644, 2024-02) | A small AR model attends to a large model's block-mode representations. | 1.16× vs PaLM2-Otter; 1.14× in SPEED. | Yes. | adj / none / none |

## 2. Cross-layer maps, lenses and interpretability: tools for (C), and the "ridge part" of the bridge

| Paper | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **Jump to Conclusions** (Din, Karidi, Choshen, Geva; arXiv 2303.09435, LREC-COLING'24) | Learned *linear* maps from layer-ℓ hidden states to final-layer space; can also linearly approximate sub-modules. | Better than reading intermediate layers directly. Extra 7.9% (GPT-2) / 5.4% (BERT) layers saved at 95% accuracy. Attention is the most linearizable sub-module. | Linear maps. | **adj** (same-token cross-depth linear map; our ridge stage is cross-*token*) / none / none |
| **Tuned Lens** (Belrose et al.; arXiv 2303.08112) | Per-layer affine probe on a frozen model, decoded to the vocabulary. | More predictive than logit lens; tested up to 20B. | Probes only. | adj (tool for C) / none / none |
| **Future Lens** (Pal et al.; arXiv 2311.04897, CoNLL'23) | Linear approximation and causal intervention decode tokens ≥2 positions ahead from one hidden state (GPT-J). | >48% accuracy at some layers. | Linear. | **adj** (cross-position state prediction) / none / none |
| **Patchscopes** (Google; arXiv 2401.06102, ICML'24) | Patch a hidden state into another prompt or layer and let the model decode it; unifies lens methods. | Fixes the early-layer blind spots of lenses. | No. | adj (C: parent→child patching tests) / none / none |
| **Layer by Layer** (Skean et al.; arXiv 2502.02013, ICML'25 camera-ready per comments) | Representation-quality metrics show intermediate layers beat the final layer on 32 embedding tasks. | Analysis. | No. | adj (C) / adj (history encoding at a mid layer) / none |
| **Remarkable Robustness / Stages of Inference** (Lad, Lee, Gurnee, Tegmark; arXiv 2406.19384) | Delete or swap adjacent layers: 72–95% of top-1 retained. Proposed stages: detokenization → feature engineering → prediction ensembling → residual sharpening. | Early and final layers are fragile; the middle is robust. | No. | adj (C) / adj / none |
| **Transformer Layers as Painters** (arXiv 2407.09298, AAAI'25†) | Middle layers share a representation space; skipping, reordering or parallelizing middle layers on frozen models. | First and last layers are critical. | No. | adj (C) / none / none |
| **ReplaceMe** (arXiv 2505.02819, NeurIPS'25 per comments) | **Training-free:** replace a contiguous span of blocks with *one linear transform* estimated on calibration data, then merge it into weights. | Prune up to 25% of blocks while keeping ~90% of performance. | No (calibration only). | **adj** (closest to our "ridge" stage, but same-token) / none / adj |
| **LLM-Streamline** (arXiv 2403.19135, v5 2025-01) | Prune consecutive low-importance layers and train a lightweight network to replace them. | SOTA among pruning methods (no numbers on abstract). | Lightweight replacement network. | **adj** (light learned module replaces a layer span; same token) / none / adj |

## 3. Prompt-side compute reduction, cross-layer KV and early-exit KV

| Paper | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **SwiftKV** (Snowflake; arXiv 2410.03960, EMNLP'25) | **SingleInputKV:** for *prompt* tokens, every layer j > l gets KV_j = W_KV^j · x_{l+1} from layer l's output. Prompt tokens skip attention and MLP above l. Decode tokens and the last prompt token run all layers. **AcrossKV** shares KV across adjacent layers. | Llama-3.1-8B-Instruct average: 73.71 baseline → 73.59 at 25% → 72.70 at 50% → 66.09 at 62.5% of layers skipped. Up to 2× throughput, −60% TPOT. Motivated by cosine similarity >0.5 between mid and deeper inputs. | **Only the Q, K, V of layers > l** (<10% of params), logit distillation T=2, ~680M tokens, <3 h on 8×H100. | adj (inverse of LE) / **DIRECT** (the early-exit-KV half of idea A) / **DIRECT** (light-FT recipe: train only upper QKV) |
| **KV Prediction** (Apple; arXiv 2410.08391, 2024-10) | A small auxiliary model processes the prompt and *predicts the base model's KV cache*; the base decodes on it. | +15–50% relative TriviaQA accuracy at fixed TTFT FLOPs; real TTFT gains on M2 CPU. | Yes (auxiliary model + predictor). | adj / **DIRECT** (cheap history KV) / adj |
| **YOCO** (MSR; arXiv 2405.05254, NeurIPS'24†) | Self-decoder produces one global KV; the cross-decoder reuses it. **Prefill can exit before the cross-decoder** without changing output. | Order-of-magnitude memory and prefill gains (no numbers on abstract). | Yes (from scratch). | none / **DIRECT** (architecture form of "history only needs lower half") / adj |
| **LCKV** (Wu & Tu; arXiv 2405.10637, ACL'24) | Queries of all layers attend to the KV of only a few (top) layers. | Up to 26× throughput. | Yes. | none / adj / adj |
| **CLA** (arXiv 2405.12981, NeurIPS'24†) | K/V heads shared across adjacent layers. | 2× further KV cut vs MQA at near-equal accuracy (1B/3B). | Yes (from scratch). | none / adj / none |
| **Systematic Study of Cross-Layer KV Sharing** (arXiv 2410.14442, NAACL'25) | Unified framework (LCKV, YOCO, CLA, variants). | At high reduction ratios, the best variant pairs queries of all layers with upper-layer KVs (needs extra training, higher prefill latency). | Yes. | none / adj / adj |
| **KVSharer** (arXiv 2410.18517) | Plug-and-play: share *dissimilar* layers' KV. | −30% KV compute; ≥1.3× generation. | No (calibration). | none / adj / none |
| **Gemma 3n KV-cache sharing** (Google developer blog, 2025-06) | Middle-layer local and global KV is *shared to all top layers* during prefill. | ~2× prefill vs Gemma 3 4B (vendor claim). | Trained-in. | none / **DIRECT** (production precedent for "upper layers reuse a mid-layer KV") / adj |
| **GemFilter** (arXiv 2409.17422) | Early layers act as a filter to select prompt tokens; then the full model runs on the compressed input. | 2.4× speedup, −30% memory; beats SnapKV on NIAH. | No. | none / adj (early layers decide relevance) / none |
| **FastKV** (arXiv 2502.01068, Findings ACL'26 per comments) | Full context up to a Token-Selective Propagation layer, then only selected tokens propagate. KV budget decoupled. | 1.82× prefill, 2.87× decode. | Not stated. | none / adj / none |
| **LazyLLM** (Apple; arXiv 2407.14057) | Dynamic prompt-token pruning in prefill and decode; pruned tokens can be revived. | 2.34× prefill on Llama-2-7B multi-doc QA. | No. | none / adj / none |
| **SnapKV** (arXiv 2404.14469, NeurIPS'24†) | Observation window picks per-head important prompt KV. | 3.6× generation, 8.2× memory at 16K. | No. | none / adj / none |
| **FastGen** (arXiv 2310.01801, ICLR'24) | Per-head profiling to decide which KV to keep. | Large memory savings, negligible loss. | No. | none / adj / none |
| **LayerKV** (arXiv 2410.00428) | Layer-wise KV block allocation and offload, SLO-aware scheduling (systems). | Up to 69× TTFT. | No. | none / none / none |

## 4. Position-independent caching, block attention and parallel encoding

| Paper | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **Prompt Cache** (arXiv 2311.04934, MLSys'24) | Precompute attention states of reusable "prompt modules" defined by a schema that fixes positions. | ~8× TTFT (GPU), up to 60× (CPU). | No. | none / adj / none |
| **Parallel Context Windows** (arXiv 2212.10947, ACL'23) | Chunks attend only within their window; position IDs reused across windows. | ICL gains from 750M to 178B. | No. | none / **DIRECT** (training-free block-diagonal context) / none |
| **Block-Attention** (arXiv 2409.15355, ICLR'25) | Each block (passage) computes KV independently with position re-encoding; the last block attends to all. | −98.7% TTFT, −99.8% FLOPs; parity on 11 benchmarks. | **Yes, block fine-tuning.** | none / **DIRECT** (the history-item version of this is idea A) / adj |
| **TurboRAG** (arXiv 2410.07590) | Offline per-chunk KV, examined masks and position embeddings. | 8.6× average (9.4× max) TTFT. | Yes (fine-tune). | none / DIRECT / adj |
| **CacheBlend** (arXiv 2405.16444, EuroSys'25†) | Reuse non-prefix chunk KV; selectively recompute a small token subset to restore cross-attention; pipelined with loading. | 2.2–3.3× TTFT, 2.8–5× throughput. | No. | none / DIRECT (training-free recovery baseline) / none |
| **EPIC / LegoLink** (arXiv 2410.15332, ICML'25†) | Position-independent caching; LegoLink targets each chunk's attention-sink tokens. | Up to 8× TTFT, 7× throughput. | No. | none / DIRECT / none |
| **APE** (arXiv 2502.05431, ICLR'25) | Parallel encoding aligned to sequential via shared prefix, attention temperature and scaling factor. | 98% (RAG) / 93% (ICL) of sequential; 4.5× end-to-end. | No (inference-time). | none / DIRECT (training-free alignment knobs) / none |
| **KVLink** (arXiv 2502.16002, v4 2025-11) | RoPE re-encoding of each cached doc, plus trainable *link tokens* that restore cross-doc attention. | +4% QA vs prior PIC; −96% TTFT. | Yes (link tokens + FT). | none / DIRECT / adj |
| **KV Packet** (arXiv 2604.13226, NeurIPS'26 per comments) | Cached docs become immutable "packets" wrapped in trainable soft-token adapters, trained by self-distillation. **Zero recomputation.** | Near-zero FLOPs; F1 ≈ full recompute; beats CacheBlend, EPIC, SAM-KV. | Adapters only. | none / **DIRECT** (frozen-model distilled fix for context-free chunks) / adj |
| **MEPIC** (arXiv 2512.16822) | Paged chunk KV; RoPE fused into the kernel so blocks are shareable across requests. | Up to 2× less HBM (5× long prompts). | No. | none / adj (systems) / none |
| **You Need an Encoder for Native PIC (COMB)** (arXiv 2602.01519, 2026-02) | Adds an 8-layer encoder plus cross-attention, comb-interleaved, to a *frozen* 32-layer decoder. The PIC cache stores cross-attention K/V per document. | −51–94% TTFT, ~3× throughput, comparable accuracy. | Yes. Encoder + cross-attn (~3.5B params, ~3k A100-h). | none / **DIRECT** (decoder-only → encoder+decoder for cacheable context) / adj |
| **Star Attention** (NVIDIA; arXiv 2411.17116, ICML'25) | Blockwise-local context encoding across hosts, then global attention for the query. | Up to 11× less memory and time at 97–100% accuracy. | No. | none / adj / none |
| **CEPE** (arXiv 2402.16617, ACL'24) | Small encoder encodes chunks in parallel; frozen decoder uses them through inserted cross-attention. | 128K context, ~10× throughput, 1/6 memory. | Encoder + cross-attn. | none / DIRECT (encoder bolted onto a frozen decoder) / adj |

## 5. Decoder-only → encoder(-decoder), and decomposed lower/upper layers

| Paper | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **DeFormer** (Cao et al.; arXiv 2005.00697, ACL'20) | Lower layers use *question-wide and passage-wide* self-attention only, so the passage is precomputed. Upper layers are joint. Distillation-based losses. | >4.3× speedup, ~1% accuracy drop (BERT/XLNet QA). | Yes (init from pretrained, distilled FT). | none / **DIRECT** (exactly idea A's encoder-lower/joint-upper split, for encoders) / adj |
| **PreTTR** (MacAvaney et al.; arXiv 2004.14255, SIGIR'20) | Precompute document term representations up to layer l at index time; join with the query in upper layers. Trained compression cuts storage up to 95%. | Up to 42× lower query latency. | Yes (ranker FT + compression). | none / **DIRECT** (IR analogue: per-item lower-layer cache) / adj |
| **Deep Encoder, Shallow Decoder** (Kasai et al.; arXiv 2006.10369, ICLR'21) | AR MT with a deep encoder and a 1-layer decoder matches NAT speed with better quality. | Speed/quality Pareto. | Yes. | adj (justifies a few upper layers for SIDs) / adj / adj |
| **Encoder-Decoder Gemma** (Google; arXiv 2504.06225, 2025-04) | Adapts pretrained Gemma 2 decoder-only (2B/9B) into encoder-decoder (incl. unbalanced 9B-2B). | Comparable or better pretraining at equal inference; +7% after IT (2B-2B). | Yes (adaptation pretraining). | none / adj (needs heavy training) / adj |
| **T5Gemma 2** (Google blog / tech report, 2025-12) | UL2 adaptation of Gemma 3 to encoder-decoder; tied embeddings; merged self+cross attention. | 270M–4B sizes, 128K context. | Yes. | none / adj / adj |
| **LLM2Vec** (arXiv 2404.05961, COLM'24) | Bidirectional attention + masked next-token prediction + contrastive turns a decoder into an encoder. | SOTA unsupervised MTEB. | Light. | none / adj (item-local bidirectional encoding is possible) / none |

## 6. Depth pruning, general and recsys

| Paper | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **ShortGPT** (arXiv 2403.03853) | Block Influence (1 − cos(in, out)) ranks layers; remove the lowest. | Beats prior pruning (abstract has no numbers). | No. | adj (layer redundancy) / none / adj |
| **Unreasonable Ineffectiveness of the Deeper Layers** (Gromov et al.; arXiv 2403.17887, ICLR'25 per v2) | Angular-distance block selection; deeper blocks pruned. | Minimal degradation up to ~half the layers (QA benchmarks). | QLoRA "healing". | adj / none / adj |
| **LaCo** (arXiv 2402.11187, Findings EMNLP'24) | Collapse rear layers into a prior layer. | >80% average performance at 25–30% pruning. | No (post-training explored). | none / none / adj |
| **SLEB** (arXiv 2402.09025, ICML'24) | Remove redundant transformer blocks by verification. | Better speedup and perplexity than prior pruning. | No. | none / none / adj |
| **Minitron** (NVIDIA; arXiv 2407.14679) and **Minitron in practice** (arXiv 2408.11796) | Depth, width, attention and MLP pruning plus distillation (<3% of original data). | 15B→8B/4B; up to 16% MMLU over scratch. | Yes (KD). | none / none / adj |
| **What Matters in Transformers? Not All Attention Is Needed** (arXiv 2406.15786) | Similarity-based drop of attention layers (and of attention+MLP). | Llama-2-70B: drop half the attention → +48.4% speed, −2.4%. | No. | adj / adj / adj |
| **Reassessing Layer Pruning** (arXiv 2411.15558; ICLR'26 per ledger†) | Pruning the *last 25%*, then fine-tuning lm_head and the last three layers, beats fancy metrics. | Llama-3.1-8B-It pruned beats several 6–7B models. | Light FT. | adj / none / **DIRECT** (cheap FT of last layers is enough) |
| **SLMRec** (arXiv 2405.17890, ICLR'25) | Motivating study: most intermediate LLM layers are redundant for sequential rec (8-layer ≈ 24-layer). Distil into a small model. | 13% of params; up to 8× inference speedup. | Yes (KD). | adj (C: rec needs few layers) / none / adj |
| **Pruning LLM-based rec** (Zheng, Bao, Zhang et al.; arXiv 2507.07064) | Intra-layer (attention/MLP) + layer-wise pruning, each stage followed by distillation. | ~88% of performance with >95% of non-embedding params removed. | Yes. | adj / none / adj |

## 7. Recsys: KV reuse, history caching, item-local encoding, late interaction, GR decoder shortcuts

| Paper (org, venue, date) | Mechanism | Result | Training? | LE / A / B |
|---|---|---|---|---|
| **Bat: Bipartite Attention** (ZJU + Alibaba Taobao; ASPLOS'26, DOI 10.1145/3779212.3790131) | "Item-as-prefix": candidate items' KV are precomputed **per item, context-free**. Block masks keep items from attending to each other; all items share the same start position ID. The user and instruction attend to cached item KV. Disaggregated user/item KV pool; hot-replicated, cold-sharded items. **Base: Qwen2-1.5B (28 layers), also Qwen2-7B and Llama3-1B.** | Up to 1.6× throughput vs user-as-prefix; −58% compute.<br>Quality (ranking, Recall@10): IP vs UP mixed, e.g., Beauty 0.683 vs 0.656, Books 0.552 vs 0.576. CacheBlend-style PIC narrows the gap. | Fine-tuned rankers; the authors say position encoding can be adjusted in periodic FT. | none / **DIRECT** (per-item context-free KV; but for candidates, all layers, not history-lower-layers) / adj |
| **RcLLM** (HKBU/Amelie Zhou group; arXiv 2605.07443, ICDCS'26) | Per-item KV blocks for *candidate items* (offline, by item ID; METIS-sharded). *History* tokens are matched to ~10^5 LSH prototypes with precomputed KV (~30 GB for Qwen3-8B). Layer-1 full attention scores tokens; heavy hitters plus a local window are recomputed exactly (CacheBlend-like, ratio 0.2–0.5). | P50 TTFT 1.3–2.2× (simulated, Vidur); peak 9.5×.<br>Quality: Amazon NDCG@10 0.403→0.438, but **Yelp 0.477→0.407, Goodreads 0.427→0.393**. | No (precompute only). | none / **DIRECT** (training-free history/item KV reuse; quality loss visible) / none |
| **EARN** (Yang, Lin, Wang et al.; arXiv 2507.00715, KDD'25) | Prefix+suffix *register tokens*. Layers 1..k see everything; **above layer k the prompt (instruction+history) tokens are removed**, so upper layers see only registers and generated tokens. k≈L/4 (Llama k=4, Qwen k=7). Reports "attention-sparsity inversion" in LLMRec (dense early, sparse late) and dual sinks. | 2.71–3.79× wall-time; −66.7–80.8% KV; ≥ FT accuracy (LC-Rec and TIGER on Llama-7B and Qwen2.5-7B). Up to 7× at 20K tokens. | **Full fine-tuning** (lr 1e-3). | adj / **DIRECT** (history exits early; but drops instead of SwiftKV-projecting, and needs full FT) / adj |
| **RelayGR** (Huawei Ascend, 41 authors; arXiv 2601.01712, 2026-01) | Pre-infer the user-behavior prefix KV during retrieval; keep it in HBM for ranking. Sequence-aware trigger, affinity router, DRAM expander. | Up to 1.5× longer sequences and 3.6× SLO-compliant throughput. | No. | none / adj (prefix reuse) / none |
| **MTServe** (arXiv 2604.22881, 2026-04) | Hierarchical GPU/host-RAM user-KV cache; async transfer; locality-driven replacement. | Up to 3.1×; >98.5% hit ratio. | No. | none / adj / none |
| **RACER: When KV Meets Embeddings** (arXiv 2605.04450, SC'26) | PPO controller splits HBM between KV and embedding caches; KV-residency-aware routing. | P99 −24–38%. | Controller. | none / adj / none |
| **CollectiveKV** (arXiv 2601.19178, ICLR'26) | Learnable global KV pool shared across users plus low-dim user-specific KV (from an SVD finding). | KV down to 0.8% of original at equal or better quality. | Yes. | none / adj / none |
| **HSTU / M-FALCON** (Meta; arXiv 2402.17152, ICML'24) | Candidates in microbatches; masks make candidates attend to history but not to each other; history KV cached across microbatches and requests. | 285× more complex model at 1.5–3× throughput; QPS 1.50× (1K cands), 2.99× (16K). | Trained. | none / adj (history computed once, candidate-local) / none |
| **OneTrans** (ByteDance+NTU; arXiv 2510.26104, WWW'26) | **Pyramid stacking:** at each layer only the tail of the most recent sequence tokens issues queries (1190→12). K/V still cover the full sequence. **Cross-request KV caching:** sequence-side K/V computed once per request, and append-only incremental across requests (O(ΔL)). | Cross-request caching: −29.6% p99 inference latency, −52.9% memory; +5.68% GMV/user online. | Trained. | none / **DIRECT** (history queries "exit" in upper layers) / adj |
| **LONGER** (ByteDance; arXiv 2505.04421, RecSys'25) | Global tokens, token merge with InnerTransformers, hybrid attention, KV-cache serving. | Deployed in >10 scenarios. | Trained. | none / adj / none |
| **PinFM / DCAT** (Pinterest; arXiv 2507.12704) | 20B+ user-sequence foundation model; Deduplicated Cross-Attention Transformer for serving. | +600% throughput; +20% new-item engagement. | Pretrain + FT. | none / adj / none |
| **MARM** (Kuaishou; arXiv 2411.09425, CIKM'25) | Memory-augmented rec: caches intermediate computation to trade cache for FLOPs ("cache scaling laws"). | — | Trained. | none / adj / none |
| **OneRec-V2 lazy decoder-only** (Kuaishou; arXiv 2508.20900) | Context is mapped to fixed-dim features. A "Context Processor" splits them into per-layer K/V with **no transformer layers over context**. Decoder blocks cross-attend with Q only; K/V shared across all layers (L_kv=1, v=k), GQA G_kv=1. | 1B: 18.9 vs 296.4 GFLOPs (−94%); loss unchanged (3.27). | Trained (scaling 0.1–8B). | adj / **DIRECT** (extreme "history = cheap KV") / adj |
| **HatLLM** (ZJU; arXiv 2510.10955, 2025-10) | Layer-wise masks: **shallow layers intra-item only** (4 or 8 layers best), middle full causal, deep layers (1–3) cross-item via each item's last token. Llama3-3B (1B/8B also). | +9.13% average over the best LLM-rec baseline. No efficiency claims. | LoRA r=8. | none / **DIRECT** (evidence item-local lower layers *help* quality) / adj |
| **HLLM** (ByteDance; arXiv 2409.12740) | Item LLM encodes each item's text independently; a user LLM runs over item embeddings. | SOTA on PixelRec/Amazon; scales to 7B. | Yes. | none / adj (context-free item encoder) / none |
| **LEARN** (Kuaishou; arXiv 2405.03988, AAAI'25) | Frozen LLM item encoder (embeddings precomputable) plus a trained twin tower. | Deployed. | Towers only. | none / adj / none |
| **MixLM** (LinkedIn; arXiv 2512.07846, v2 2026-01) | Catalog items encoded nearline into a few embedding tokens, cached, and mixed with text in the ranker LLM. | 10× throughput vs summarized-text and 75.9× vs full-text rankers at equal latency; +0.47% DAU. | Yes. | none / adj (item compression, not KV reuse) / none |
| **Semantic Search at LinkedIn** (arXiv 2602.07309, 2026-02) | Prefill-oriented ranking stack: pruning, context compression, text-embedding hybrid; distilled SLM. | >75× ranking throughput at a fixed latency SLO. | Yes. | none / adj / none |
| **RASTP** (arXiv 2511.16943, WWW'26 short) | Prunes low-saliency SID tokens (representation magnitude + attention centrality) at an intermediate layer. | −26.7% *training* time. | Training-time. | none / adj (history SID tokens are prunable mid-depth) / none |
| **Deco** (Gusak, Oseledets, Frolov; arXiv 2610.32980, v2 2026-10) | History items as single fused tokens; backbone computes a next-item "concept" once; a light KV-cached decoder runs the beam. | 3.2–8.6× throughput at p95 ≤ 50 ms; equal or better accuracy. | Yes (CE ± distillation). | adj (shallow SID decoder) / adj / adj |
| **SID-MLP** (UCSD+Snap; arXiv 2605.12617) *(ledger)* | Distil the AR SID decoder into per-position MLPs over a once-computed user context. **SID difficulty drops sharply after the first token.** | 8.74× on A6000 (TIGER-scale teacher). | Yes. | adj (motivation for LE) / none / adj |
| **NEZHA** (Alibaba; arXiv 2511.18793, WWW'26) *(ledger)* | Self-drafting head inside the LLM plus hash-set verification. | 48 ms → 4–5 ms (Qwen); deployed. | Yes. | adj / none / none |
| **PrefixMem: "LLMs Need Encoders for Semantic IDs Too"** (Pinterest data; arXiv 2606.00324, 2026-05) | Prefix n-gram memory tables give prefix-conditioned representations at SID positions (encoder for SIDs, like a vision encoder). | +46% relative deepest-level SID accuracy; +22% recall. | Yes (pretrained, then joint). | adj (SID positions benefit from prefix-conditioned states, as in our bridge) / none / adj |
| **Gryphon** (arXiv 2606.08604, 2026-06) | Encoder-decoder SID generation plus an item-level scorer reusing the encoder's user representation. | +3.7% Recall@1000. | Yes. | none / none / none |
| **ILRec** (arXiv 2602.17410, 2026-02) | Self-hard negatives from *intermediate layers* of LLM recommenders for preference optimization and distillation. | Gains (abstract). | Yes. | adj (C: intermediate-layer item predictions are informative) / none / none |

Not individually verified (seen only in search snippets):
- IMFuse (arXiv 2607.27002): intermediate layers capture item semantics at different granularities; deeper layers show spectral collapse.
- HiLaR (arXiv 2607.27760): layer-wise alignment to quantized preferences.
- OptiRAG-Rec (ACL'25): multi-head early exit for LLM CTR.
- InferScale (arXiv 2607.27090): chunked-RoPE injection of precomputed user KV.
- jina-reranker-v3 / BReps: rerankers with precomputed document representations.

They are relevant to (C) and A but below the bar for the tables.

---

## 8. Analysis

### 8.1 Late-entry bridge vs. closest priors: precise differences

Our operator is h_e(child) ≈ B(h_e(parent), E[child], E[parent]):
- B is ridge + a 2-layer MLP, KL-distilled to the full-depth next-level distribution.
- The child **never computes layers 0..e−1**. It runs layers e..27 with exact attention to the prompt KV and the parent KV at those layers.
- The base model is frozen. The output is used **without verification**.

| Prior | Same | Different |
|---|---|---|
| GR4AD LazyAR | Previous-SID dependency enters only at an intermediate layer; the lower stack is shared across beams; the rec-specific rationale (later SID levels are easy, beams are wide). | Trained from scratch (9 layers) with an MTP auxiliary loss. Lower layers for position *t* run on a *token-agnostic* trunk (position embedding + context) and the parent token is fused at K. We *predict* the child's mid-layer residual from the parent's state with a post-hoc bridge on a frozen 28-layer pretrained LLM. LazyAR reports a "marginal" revenue drop; we are lossless on HR/NDCG. |
| FIRP / Hidden Transfer | Cross-position prediction of a layer-*t* hidden state from earlier positions' layer-*t* states; the injected state then runs the remaining layers with attention to context; KL to the true distributions; the best entry layer is mid-depth (15/32). | They predict *unknown* future tokens (draft → tree verification). We condition on the *known* child token (scoring trie candidates), so the map is much lower-entropy. Theirs is linear per step; ours is nonlinear. No verification pass. A hierarchical tree with many children per parent. |
| EAGLE (1/2/3) | Inputs are the parent feature plus the child token embedding, giving the child feature. | Top (penultimate) layer and a separate 1-layer draft decoder; verification needed. We enter at a *middle* layer and let the real upper layers correct errors, which is why no verification is needed. |
| SkipDecode | Later tokens skip lower and middle layers and attend to earlier tokens' upper-layer KV; no KV holes. | The entry state is the token's *own* 1-warm-up-layer output, and the whole model is fine-tuned. Ours is parent-conditioned, distilled and frozen. SkipDecode degrades past 2×; ours is lossless at a fixed e. |
| SwiftKV / CALM / Jump-to-Conclusions / ReplaceMe / LLM-Streamline | Light learned maps substitute for skipped layers, distilled to the teacher. | All are *same-token* maps (or upward KV projection for exited tokens). Ours is a *cross-token, cross-depth* substitution, which only works because the child token is drawn from the parent-conditioned SID codebook. |

**Why there is no KV hole.** SID nodes are the last tokens of the sequence. Nothing ever attends to a child below its entry layer, as long as grandchildren enter at e' ≥ e. CALM, D-LLM, SkipDecode and ADEPT all spend machinery on filling holes that we do not have.

**Claim we can defend:** the first *post-hoc*, frozen-model late entry for LLM generative recommenders, using a distilled parent→child residual bridge, lossless on ranking metrics. Cite GR4AD LazyAR as the native, trained counterpart, and FIRP and EAGLE as the general mechanisms.

### 8.2 Idea A in recsys: done vs. open

**Done:**
- Per-item, context-free, all-layer KV for *candidate* items with block masks and shared start positions. Bat does this on Qwen2-1.5B, fine-tuned, with small mixed quality changes. RcLLM does it training-free, with selective recompute and visible NDCG loss on 2 of 3 datasets.
- User-prefix KV reuse across stages and requests: RelayGR, MTServe, OneTrans, M-FALCON, PinFM, LONGER.
- History tokens removed above layer k ≈ L/4 with register tokens and full fine-tuning (EARN).
- Upper-layer history *queries* pruned (OneTrans pyramid).
- Context as K/V-only, trained from scratch (OneRec-V2).
- Intra-item-only shallow layers improve quality under LoRA (HatLLM).
- Outside recsys, idea A is DeFormer/PreTTR (2020) plus Block-Attention, KVLink, KV Packet and COMB for decoder LLMs.

**Open (no paper found):**
1. **Item-local lower layers for *history* SID tokens in a frozen GR LLM**, with a per-item cache.
   - The re-positioning is exact under RoPE: within-item attention depends only on relative offsets, so the pre-RoPE K can be rotated exactly to any absolute position. The only approximation is the mask change itself.
2. **SwiftKV-style history early exit in GR**: upper-layer K/V projected from layer m.
   - EARN drops the tokens instead and needs full fine-tuning.
   - SwiftKV's recipe trains only the upper Q/K/V (<10% of parameters) and fits our "frozen + light" constraint.
3. **(1)+(2) combined.** The history KV becomes a pure per-item lookup across *all* layers, so prefill needs only the template tail plus the SID nodes.
   - Sizing: Qwen2-1.5B KV is 28,672 B per token (Bat's table), so 86 KB per 3-token item. A catalog of ~10^4 items fits in about 1 GB.
4. **A distilled *link* correction** (KVLink, KV Packet style soft tokens), instead of CacheBlend recompute, in recsys.
5. **Honest caveat:**
   - At batch 1, prefill of about 120 tokens on a 1.5B model is weight-read bound, so idea A mostly buys *throughput* (batched or concurrent serving) and memory, not single-request latency.
   - It becomes a latency win only if it removes a *sequential pass*, or if histories are long. The industrial setting has longer histories than Office.

### 8.3 Idea B (deeper or native late entry with light fine-tuning)

The available recipes, from lightest to heaviest:
1. Train only the upper-layer Q/K/V on late-entry inputs with logit KD (SwiftKV: <10% of parameters, under 1B tokens).
2. Train the bridge and LoRA on layers e..27 jointly, with "training-time test" (EAGLE-3): train on the model's own bridged states, including the multi-hop c←b←a chain.
3. LayerSkip-style layer dropout plus an early-exit loss restricted to SID positions.
4. A LazyAR-style MTP auxiliary loss that makes the shallow trunk predictive.

Evidence that the last few layers can be cheaply re-fitted: "Reassessing Layer Pruning" (fine-tuning the last layers plus the head suffices) and SLMRec (rec needs few layers).

Negative evidence to cite:
- Learning to Skip the Middle Layers: learned skipping did not beat dense baselines.
- FlexiDepth: no wall-clock gain from per-token skipping.

Both point the same way: do **static, per-level** entry depths, not per-token routers.

### 8.4 Tools for (C), "where is the item decided"

- Per-layer tuned lens, or Jump-to-Conclusions linear maps, on SID logits for each level (a, b, c).
- Future Lens style probes: can the layer-ℓ state of the parent predict the child?
- Patchscopes-style parent→child patching. Our bridge is in effect a learned patch; sweep *e* and report KL and HR.
- Attention-to-history by layer. EARN reports sparsity inversion in LLMRec; check it on MiniOneRec, since it determines where history can exit.
- SwiftKV's cosine-similarity-across-depth plot.
- Stages-of-inference framing (detokenize → features → ensemble → sharpen).

---

## 9. Ranked experiments (novel, large speedup, in our setting)

**E1. Depth-staggered single-pass trie scoring.** Highest expected batch-1 gain, lowest quality risk.
- **Mechanism:**
  - Every bridge input lives at layer *e* and needs no upper layer. The prompt-last state at *e* feeds the *a* nodes; an *a* node's bridged state at *e* feeds its *b* children.
  - So inject SID nodes into the prompt's own forward pass when it reaches layer *e*, and run layers e..27 once over {prompt tail, a-nodes, (a,b)-nodes} with a tree mask.
  - That gives the a, b and c distributions from **one** sequential pass instead of three, with *no new approximation* relative to the current late-entry model.
  - If the (a,b) set is too large, use two passes instead:
    1. Prompt + all ≤256 a-nodes; read the top-K_a.
    2. One tree pass for (a,b) under the top-K_a. This also makes the certificate exhaustive inside the frontier.
- **Prior to cite:** FIRP injects pseudo-states mid-pass, but for drafting with verification.
- **Risk:** low on quality.
  - Medium on cost: thousands of nodes × (28−e) layers can become compute-bound.
  - Needs a tree-mask Triton kernel. The existing `--tree-fast` prompt-attention kernel is a starting point.
  - First check whether the current engine already fuses the b and c passes.

**E2. Deeper entry via SwiftKV-style light FT (idea B).**
- **Mechanism:** freeze everything except the upper-layer Q/K/V (and the bridge). Distil on validation prompts so that e moves from 18–22 to 24–26.
- **Payoff:** the per-level upper stack shrinks from 6–10 layers to 2–4, roughly 2× fewer decode-layer weight reads on every non-first level. It compounds with E1.
- **Risk:** medium. It breaks "frozen model", but it is <10% of parameters and reversible. Validate Office and Industrial HR/NDCG plus teacher top-K overlap.

**E3. History as a per-item lookup (idea A).**
- **Steps:**
  1. Lower m layers item-local: block-diagonal per 3-token item, exact RoPE re-rotation.
  2. Upper-layer history KV from a SwiftKV projection of layer m.
  3. Precompute the KV of every catalog item for all 28 layers.
  4. Recover quality with a distilled fix (KVLink/KV Packet-style soft link tokens, or upper-QKV FT).
- **Ablations:**
  - m ∈ {2, 4, 8, 14}.
  - Training-free first: APE temperature, CacheBlend recompute of the top-r tokens.
- **Risk:** medium-high on quality, but HatLLM and EARN suggest history information is consolidated early.
- **Payoff:** mostly throughput and TTFT at batch > 1 and with long industrial histories. Report it as throughput, not batch-1 latency.

**E4. Layer-wise "where is the item decided" atlas (C).** Low risk; it supports the paper narrative.
- **Measure:**
  - Per-level tuned-lens and ridge-probe accuracy by layer.
  - Bridge KL/HR vs *e* for each level.
  - Attention mass on history vs. template vs. parent by layer (test EARN's inversion claim).
  - Hidden-state cosine across depth.
- **Output:** a principled choice of per-level e_b and e_c (e_c > e_b is likely, since c is easier per SID-MLP), plus the history exit layer m for E3.

**E5. Deep-first cascaded certification.**
- **Mechanism:**
  1. Score all trie nodes with a *deeper* bridge (e.g., e=25, cheap).
  2. Re-score only the nodes whose score is within a margin of the top-K boundary at the validated e=18–22.
  3. Certify exactness w.r.t. the e=18 model when the margin bound holds.
- **Payoff:** this is the LayerSkip/Draft&Verify idea, applied to scoring rather than drafting.
- **Risk:** low-medium. The speedup depends on the margin distribution and needs a bound on |score_25 − score_18|. That bound can be estimated empirically on validation.

**Not recommended:**
- Per-token routers (MoD, D-LLM, FlexiDepth style), which show no wall-clock gains at batch 1.
- Early exit for level *a*: already negative in our runs, and consistent with Dynamic Layer Selection's finding that early exit is fragile.
- Heavy encoder-decoder conversion (Encoder-Decoder Gemma, COMB, ~3k A100-h). Out of budget, and not needed given E3.
