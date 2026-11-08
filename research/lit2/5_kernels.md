# 5. Kernels: from 2.9 ms toward the ~1 ms floor on H100, and more throughput

*Sweep date: 2026-11-08. Target: Qwen2-1.5B FP16 (28 layers, d=1536, 12q/2kv heads, MLP 8960) on one H100 SXM (3.35 TB/s HBM3, 132 SMs, 50 MB L2). A6000 notes are included where they differ. Numbers without a tag come from the cited source. Numbers tagged **(est.)** are my arithmetic or extrapolation for our shapes; benchmark them before relying on them.*

This file complements `research/kernel_techniques.md` (2026-11-07). That file was written for the older A6000 beam-search engine and covers split-K, Marlin and GemLite tables, and a CUDA-graph recipe. This file is about **H100 batch-1 latency of the current tree engine** and **B=64 throughput**.

---

## 0. Budget math: where the 2.9 ms goes

**Weights per layer:** QKV (1536→2048) 6.29 MB, O 4.72 MB, gate+up 55.05 MB, down 27.53 MB, for **93.6 MB per layer** in FP16.

| Pass | Layers | Bytes | Rows (M) |
|---|---|---|---|
| Forward 1 | 0–27 | 2.62 GB | ~7–50 new prompt rows in layers 0–17; plus 88 tree rows in layers 18–27 |
| Forward 2 | 22–27 | 0.56 GB | 16–30 rows |
| **Total per request** | 34 layer-passes | **3.18 GB** | |

**Memory floor for 3.18 GB:**

| Achieved bandwidth | Floor |
|---|---|
| 3.35 TB/s (peak) | 0.95 ms |
| 3.0 TB/s (~90%, well-tuned streaming) | 1.06 ms |
| 2.61 TB/s (78%, Hazy megakernel's achieved MBU) | 1.22 ms |

A **realistic best case is about 1.2–1.4 ms**, with attention, norms and tree bookkeeping on top.

**Ideal time per GEMM at 3 TB/s:**
- QKV: 2.1 µs. O: 1.6 µs. gate_up: 18.4 µs. down: 9.2 µs. That is **about 31 µs per layer-pass**.
- The two attention-side GEMMs are so small (4.7–6.3 MB) that their ideal time is about **the same as one kernel launch in a CUDA graph (1.3 µs on H100)**. No standalone GEMM kernel can run them at bandwidth.

**Implied overhead (est.).** 2.9 ms over 34 layer-passes is about 85 µs per layer-pass against the 31 µs ideal, so about **54 µs per layer-pass is not weight streaming**. At about 10 kernels per layer (norm, QKV, bias/RoPE/KV-store, 1–2 attention, O, add+norm, gate_up, act, down), that is **about 5 µs per kernel boundary**. Hazy Research used exactly this "optimistic 5 µs stall per kernel" estimate to explain why vLLM and SGLang reach only about 50% of H100 bandwidth on Llama-1B ([No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). We are at about 33–38% MBU. **The gap to the floor is kernel boundaries, not GEMM math.** Everything below is ranked by how much boundary time it removes.

**Throughput mode (est.).**
- Forward FLOPs are 2 × 46.8 M params × rows, per layer: about **174 GFLOP per user** with 30 prompt rows, about 226 GFLOP with 50.
- 2,500 users/s is therefore **about 435–565 TFLOP/s, or 44–57% of the H100 dense FP16 peak** (989 TF).
- FP16 GEMM-efficiency headroom is therefore modest (about 1.2–1.4×). FP8 tensor cores (1,979 TF dense) are the larger lever.

**First diagnostic (1–2 h, do before anything else).** Run `nsys` on one graph replay and record three things:
1. Sum of kernel durations against wall time. The gap is launch and bubble time.
2. Kernel count per request.
3. Achieved GB/s per GEMM: weight bytes divided by duration.

This tells you whether the 1.8 ms gap is in launch gaps (fix with PDL or megakernel) or in slow kernels (fix with tiling or fusion).

---

## 1. Megakernels and persistent kernels: what is published, with numbers

| System | What | HW / model / batch | Reported result | Usable for us? |
|---|---|---|---|---|
| **Hazy "No Bubbles"** ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [code](https://github.com/HazyResearch/Megakernels), MIT) | One persistent kernel; on-GPU interpreter of 7 fused instructions per layer; shared memory paged into 13×16 KiB pages so the next instruction's weight loads start early; global-memory counters instead of grid barriers | Llama-3.2-1B BF16, batch 1, H100 and B200 | **H100: <1 ms per forward, 78% of HBM bandwidth.** About 2.5× vLLM and >1.5× SGLang (both with CUDA graphs and torch.compile). B200: <680 µs. | Design reference only. GEMV on CUDA cores (M=1); we need tensor-core tiles for M=30–190. Repo targets H100/B200 and installs torch from the cu128 index (check Oscar's driver). Llama demo only. |
| Hazy TP megakernel ([blog](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)) | Throughput megakernel with tensor-core GEMM instructions (128×64×256 tiles) | Llama-70B, 8×H100, batch 1k–8k | +22% end-to-end throughput over SGLang. Intra-SM pipelining cut the inter-matmul gap from **10.2 µs to 3.4 µs** (+2–6%). | Shows that megakernel GEMMs with tensor cores work. Gains at large batch are modest. |
| **MPK / Mirage Persistent Kernel** ([arXiv 2512.22219](https://arxiv.org/abs/2512.22219), OSDI'26, [code](https://github.com/mirage-project/mirage), Apache-2.0) | Compiler: tile-level task graph plus events; worker and scheduler SMs (H100: 128 workers); `torch.compile(backend=MPK)` | Qwen3 0.6B–30B-A3B, A100/H100/B200, batch 1–16 | **1.0–1.7×** over SGLang/vLLM at batch 1, larger for small models and newer GPUs. Qwen3-8B on A100: 14.5 → 12.5 ms per token. | Build from source (`pip install -e .`, needs nvcc). BF16 only. Qwen3 demos, not Qwen2 bias plus custom tree mask. ForgeMegakernel measured MPK at only **20.6–75.9% MBU**. |
| Event Tensor Compiler ([arXiv 2604.13327](https://arxiv.org/abs/2604.13327), MLSys'26) | Symbolic event tensors for dynamic-shape megakernels; static schedule preferred for dense models | B200, Qwen3-32B / 30B-A3B | Batch-1 dense: up to 1.15× over vLLM. MoE: 1.20× over SGLang. Warmup 35 s vs 123–583 s. | TVM-based; no public repo found. Lesson: **use a static per-SM schedule for dense models** (the dynamic scheduler was 0.82–0.89× in their TP setting). |
| **ForgeMegakernel** ([arXiv 2609.12379](https://arxiv.org/abs/2609.12379), Sep 2026) | Agent-generated per-model megakernels, built against a 10-milestone checklist | **H100 80GB HBM3**, 14 models, 0.6B–13B | MBU 50.5–85.9% (SGLang 36.5–78.9%). **Llama-3.2-1B batch 1: 74.0% MBU, 1.37× SGLang, 1.50× vLLM, 1.52× MPK.** Qwen3-0.6B batch 8: 1.41 vs 1.58 ms per token. | No code link. Ablation (Qwen3-0.6B): removing the per-SM instruction stream drops MBU 35.7 → 26.5%; removing counter dependencies → 29.6%; removing the smem buffer pool → 34.0%. |
| Triton-distributed MegaTritonKernel ([docs](https://triton-distributed.readthedocs.io/en/latest/getting-started/megakernel/megakernel.html), MIT) | Megakernel **written in Triton**: scoreboard of acquire/release flags, task queue, an L2 `prefetch` task (`cp.async.bulk.prefetch.L2` via inline asm) | Qwen3-8B/32B, TP=8, H800/H20 | Qwen3-8B on H800: **5.49 ms (torch + CUDA graph) → 3.33 ms**, 1.65×. Qwen3-32B: 10.80 → 7.41 ms. | Needs their Triton fork (thread-level `tid`, `__syncthreads`, `ld(..., semantic="acquire")`) plus NVSHMEM. The pattern ports to vanilla Triton with `tl.atomic_*(sem=, scope=)` spin-waits and `tl.inline_asm_elementwise`. |
| Cohere North-Mini megakernel ([blog](https://cohere.com/blog/megakernels), Sep 2026) | 16 opcodes, one CTA per SM, counters with `__nanosleep(20)` spin waits, tuned wave order | 30B-A3B MoE BF16, one H100, batch 1 | 292 vs 185 tok/s, **1.58× vLLM**, 62% of speed-of-light vs 39%. Dropping the tuned wave order cost 1–19%. | MoE-specific, but the lessons (wave order, weight prefetch, no false dependencies) transfer. |
| AutoMegaKernel ([arXiv 2606.09682](https://arxiv.org/abs/2606.09682)) | Agent harness plus static deadlock/race checker for schedules | TinyLlama-1.1B, A100/H100 | Its BF16 GEMV reached only ~460 GB/s (63% of peak); **trails cuBLAS** on H100 | Cautionary: auto-generated megakernels can be slower than a good library. |
| ClusterFusion ([arXiv 2508.18850](https://arxiv.org/abs/2508.18850), NeurIPS'25, [code](https://github.com/xinhao-luo/ClusterFusion)) | QKV proj + attention + O proj in **one kernel** using Hopper thread-block clusters and DSMEM (ClusterReduce / ClusterGather) | Llama-2-7B, H100, batch 1 | **1.61× average end-to-end latency** over SOTA frameworks | Exactly our "tiny QKV/O GEMM" problem. CUDA 12.4 and torch 2.5.1 in the README; Llama only. |
| TileRT ([repo](https://github.com/tile-ai/TileRT)) | Tile-level persistent runtime from the TileLang team | DeepSeek-V3.2 / GLM-5 on 8×B200 | Ultra-low-latency claims; vLLM connector (Jul 2026) | Not applicable to a 1.5B model on one H100. |

**Lessons that transfer to our tree workload:**

1. **Instruction set (Hazy).** Use 7 instructions per layer: RMSNorm+QKV+RoPE; attention; (attention reduce); O-proj+residual; RMSNorm+gate_up+SiLU; down-proj+residual; final norm+head. We currently have about 10 kernels per layer. Matching this set, even as separate kernels, is §4.
2. **Prefetch the next instruction's weights while the current one finishes.** This is the main way a megakernel beats CUDA graphs plus PDL. Hazy says PDL "still introduces stalls" because `cudaGridDependencySynchronize` is all-or-nothing, for example attention waits for all of Q, K and V.
3. **Use fine-grained counters, not grid barriers.** Hazy splits the down projection into 4 chunks with separate counters. ForgeMegakernel's ablation shows counters are worth about 6 MBU points.
4. **What remains inside a good megakernel** (Hazy, B200, 600 µs total):
   - about 250 µs of activation store/visibility/load latency (about 500 ns per round trip);
   - about 40 µs of barrier cost (about 60 ns per `mbarrier`, even when already passed);
   - about 80 µs of setup.

   Memory fences matter: "a few loose fence instructions caused over 10% loss" ([TK 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)).
5. **Our case is easier than M=1 in one way and harder in another.**
   - Easier: M=30–190 gives more work per weight byte, so tensor cores hide compute easily (arithmetic intensity ≤190 FLOP/B, still under the H100 ridge of about 295).
   - Harder: two forwards with a data-dependent tree between them, rows entering at layer 18, and a custom tree mask. Use a **static schedule per (prompt-bucket, tree-size) shape**, as ETC recommends for dense models, with padded M.

**Expected payoff for us (est.):** reaching 65–75% MBU means 1.3–1.5 ms against 2.9 ms now, about **2×**. Effort is **150–300 h** for a hand-written CUDA or Triton megakernel. It is the end state; §2–§4 are steps toward it and pay off on their own.

---

## 2. Launch overhead, CUDA graphs and PDL on Hopper

**Measured costs:**
- **Kernel launch on H100** (Hazy, dummy kernel): **2.1 µs on a stream, 1.3 µs inside a CUDA graph.** Fill/drain and tail effects come on top; for example, a 512-block grid on 148 SMs leaves 80 SMs idle in the last wave.
- **Graph launch** (CUDA 12.6, Ampere+): about 2.5 µs + ~1 ns per node for straight-line graphs ≥10 nodes ([NVIDIA](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements)). Graph launch is not our problem. The per-kernel gaps inside the graph are.

**PDL (programmatic dependent launch), sm_90+ only (not A6000).**
- **Mechanism.** The next kernel's CTAs launch while the previous grid drains. Work before `gdc_wait()` (`griddepcontrol.wait`) overlaps the predecessor, and `gdc_launch_dependents()` lets the predecessor release early ([CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)).
- **Graph support.** It works in CUDA graphs "via stream capture or directly via edge data" (native graph support since CUDA 12.3). vLLM measured it under full CUDA graphs on H200 (below).
- **Triton.** `tl.extra.cuda.gdc_wait()` and `gdc_launch_dependents()`, plus the launch kwarg `kernel[grid](..., launch_pdl=True)`. Present since **Triton 3.4.0** (PR #6394, Apr 2025; [tutorial 11](https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py)).
- **Inductor.** `torch._inductor.config.triton.enable_pdl = True`, or env `TORCHINDUCTOR_ENABLE_PDL=1`. Present since **PyTorch 2.10** (PR #160928 with "deduplicate waits" #162014; combo kernels gained PDL in #174231). Default off.
- **Measured gains:**
  - vLLM PR [#64981](https://github.com/vllm-project/vllm/pull/64981), H200, full CUDA graphs: Inductor PDL gave median TPOT **−3.1% (Qwen3-8B BF16), −4.2% (Qwen3-8B-FP8), −3.5% (Qwen3-14B-FP8) at concurrency 1**. At concurrency 8 it was +1.4% / +0.9% **worse** on the FP8 models, so it is not free at every batch size.
  - TensorRT-LLM (DeepSeek-R1, 8×B200): "Enable PDL" took 168 → 173 TPS per user (+3%) ([blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)). Their bigger use was letting a GEMM **prefetch its weights before the wait**.
- **For us (est.).** Our kernels are much shorter than an 8B model's (a 1.6–2 µs ideal O or QKV GEMM against a ~6–15 µs 8B GEMM), so launch gaps are a larger fraction of the time. Expect **−5% to −12% (0.15–0.35 ms)** if every kernel in the chain is PDL-enabled, and more if GEMMs issue their first weight-tile loads before `gdc_wait`.
- **Caveats:**
  - Any kernel launched without the attribute breaks the chain ("alternating pattern" in vLLM #40543). I found no evidence that cuBLAS/cuBLASLt GEMMs launch with PDL. Assume they break the chain, and use Triton GEMMs (Inductor `max_autotune_gemm_backends="TRITON"`, or our own) for the latency path.
  - Every PDL kernel **must** call `gdc_wait` before reading the predecessor's outputs.
  - SGLang hit graph-capture failures with its own C++ PDL launcher on sm_120 (PR #18596). Test capture on H100.

**Cheap compile-scope fixes (est. 0.05–0.15 ms):**
- **Fold the residual add into the next layer's norm.** If layers are compiled one at a time, the residual add at the end of layer *i* and the RMSNorm at the start of layer *i*+1 land in different graphs and cannot fuse. Return `(h, residual)` un-added and do `fused_add_rmsnorm` at the start of the next layer, which is vLLM's pattern. Or compile the whole stack in one region.
- **Merge independent launches.** `torch._inductor.config.combo_kernels=True` merges independent small kernels, for example the prompt-causal and tree-mask attention launches. Combo kernels also accept PDL.

---

## 3. Skinny GEMMs at M=16–224 on H100

- **cuBLAS at tiny M is decent on large matrices and poor on small ones.**
  - vLLM [#64979](https://github.com/vllm-project/vllm/pull/64979) (H200): "cuBLAS reads the weight at 2.3–3.7 TB/s at 1–4 tokens." Tuned CuTe-DSL skinny GEMMs were 1.07–1.61× faster per GEMM, giving Qwen3-8B TPOT −14.9% at batch 1.
  - **At 8 and 16 tokens the skinny kernels lost on every shape (0.4–0.9×).** For M≥16, the cuBLAS kernel itself is not the problem on 30–60 MB matrices.
  - Our problem is the **4.7–6.3 MB QKV and O GEMMs**. With 128-wide N tiles they launch only 12–16 CTAs on 132 SMs and are dominated by ramp-up and latency. Fuse them (§4) or split-K them.
- **Split-K for skinny shapes:** about **+124% on average on H100 (peak +295%)** over data-parallel tiling for m < n = k ([arXiv 2402.00025](https://arxiv.org/abs/2402.00025)). Size the split so CTAs ≈ 1–2 waves of 132.
  - Write the partials and reduce in a second step, or use a last-CTA ticket. `tl.atomic_add` split-K serialized in OKC entries; see `kernel_techniques.md` §1.
  - Put epilogues (bias, RoPE, KV store, residual, SiLU·mul) in the reduce step.
- **Swap-AB for M ≤ 64.**
  - Hopper `wgmma` has a fixed M=64 per warpgroup, and N can be 8–256. Computing Cᵀ = W·Xᵀ puts the weight dimension on the 64-wide side and the 16–64 token rows on the flexible N side.
  - vLLM [#20396](https://github.com/vllm-project/vllm/pull/20396) added swap-AB to its SM90 CUTLASS FP8 GEMM for M ≤ 64, with **5–25% gains** across shapes. FlashInfer's SM90 FP8 blockscale GEMM uses swap-AB for M < 32.
  - In Triton, express it by transposing the problem: put the weight rows on BLOCK_M and the tokens on BLOCK_N.
- **Stream-K** ([arXiv 2301.03598](https://arxiv.org/abs/2301.03598)) targets wave quantization on compute-bound shapes. The paper itself notes it does not beat cuBLAS on small bandwidth-bound shapes. Use it for B=64 throughput, not for batch-1 latency.
- **Inductor** `decompose_k` only fires when K ≫ M, N (default ratio 32), so it never fires for our shapes.
- **What to do:** use per-shape dispatch chosen offline under CUDA-graph replay with cold L2.
  - gate_up and down at M≥64: cuBLAS or Triton, whichever wins.
  - QKV and O at all M, and everything at M≤32: split-K or swap-AB Triton with fused epilogue.
  - Check that per-layer time at M=118 (tree rows present) is about equal to M=30. If not, a tile config is compute-bound at M=118 when it should not be.

---

## 4. Fusion: concrete targets and reported gains

Target: **about 4–5 launches per layer** instead of about 10. Each removed boundary is worth about 3–5 µs at our sizes, so this saves **about 0.4–0.8 ms** over 34 layer-passes (est.).

| Fusion | How | Evidence |
|---|---|---|
| **RMSNorm into the following GEMM ("lazy pre-norm")** | Fold γ into the next weight: W' = W·diag(γ), exact in real arithmetic, as QuaRot/SpinQuant do. The norm becomes affine-free: `rmsnorm(x)·W'ᵀ = rstd(x) ⊙ (x·W'ᵀ)`. Each GEMM CTA already scans the full K=1536 row of x, so accumulate Σx² alongside the k-loop and scale the output rows in the epilogue (after the reduce, with split-K). **This removes the norm kernel entirely.** | PyTorch/Meta ([blog, Jul 2026](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/)): hides **up to 90% of norm latency**. The technique needs affine-free RMSNorm, which γ-folding provides. Check FP16 range of W·γ and top-K parity. |
| gate_up GEMM + SiLU·mul | Interleave gate and up rows so one CTA's tile holds matching columns; apply `silu(g)*u` in the epilogue; write 8960-wide output (halves the activation write) | DeepFusionKernel ([arXiv 2602.11808](https://arxiv.org/abs/2602.11808)): up to **+13.2% on H100** over SGLang (fused SwiGLU MLP). Hazy instruction 5. |
| QKV GEMM + bias + RoPE + KV-cache store | Epilogue of the (split-K) QKV kernel; K and V rows go straight to cache slots, Q to the attention input | Hazy instruction 1. Standard in TRT-LLM and vLLM (fused QK-norm+RoPE also has PDL in vLLM #55755). |
| O-proj and down-proj + residual add | Epilogue; also emit Σx² of the new residual so the next lazy norm skips a pass | Hazy instructions 4 and 6. TRT-LLM's fused AllReduce+RMSNorm gave +2.4% (164 → 168 TPS). |
| QKV + attention + O-proj in one kernel | Thread-block clusters with DSMEM reduce/gather (Hopper only) | ClusterFusion: **1.61× end-to-end** at batch 1 on H100. Highest-effort item short of a megakernel. |
| Memory-bound norms done right | Cluster reduction, 128-bit loads | QuACK: about 3.0 TB/s (90% of H100 peak) for reduction dims ≥4k. Our norms are 1536-wide, so fuse rather than tune. |

All of these are pure-Triton work on Oscar. No root needed, and PDL applies to them automatically if launched with `launch_pdl=True`.

---

## 5. Quantization at our M on H100: what the evidence says

- **W4A16 and W8A16.**
  - **Marlin:** about 3.87× kernel speedup up to batch 16–32 on A10, falling toward about 1.5× by batch 128 ([arXiv 2408.11743](https://arxiv.org/abs/2408.11743)). It "struggles on H100" according to its own successor.
  - **Machete** (Hopper, vLLM ≥0.6.2, w4a16 and w8a16): beats other mixed-input kernels only at **M ≥ 128**, where it is merely "competitive with FP16". Sub-32 batch optimization was listed as future work ([Red Hat](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)).
  - **GemLite / TorchAO int4 on H100:** 1.95× at batch 1 but only 1.16× at batch 32 (Llama-3.1-8B, [PyTorch blog](https://pytorch.org/blog/accelerating-llm-inference/)).
  - **For us:** this matches the earlier "no speedup at our sizes" result for layers 18–27 (M≈118) and throughput mode. The only plausible win is **int8 W8A16 (per-channel, near-lossless) on the M≤50 passes** (layers 0–17 and forward 2, about 2.2 of the 3.2 GB) **inside the fused split-K kernel from §3–4**, where dequantization is hidden. (est.) −0.2–0.35 ms at best. Revisit only after §2–§4.
- **FP8 W8A8.**
  - **Accuracy:** "FP8 (W8A8-FP) is effectively lossless" across Llama-3.1 ([Kurtic et al., ACL'25](https://arxiv.org/abs/2411.02355)). Neural Magic's Qwen2.5-0.5B-FP8-dynamic (per-channel static weights, per-token dynamic activations) recovers 99.0% MMLU, 97.1% ARC-C and 101.3% GSM8K. Qwen2.5-1.5B INT8 W8A8 recovers 58.34 vs 58.48 average.
  - **Our −2.4 pt HR@10 used static activation scales.** Dynamic per-token scales are the first fix to try.
- **Speed on H100.** From [arXiv 2502.01070](https://arxiv.org/abs/2502.01070), Table 9 (square GEMMs, TFLOPS):
  - Rowwise with **FP32 accumulation** is slow at mid sizes: 2K gives **299 vs 840 (per-tensor)**.
  - With **fast accumulation** rowwise recovers to **810 vs 896**.
  - Thin GEMMs show "minimal improvement" on H100, for example (32, 4096, 4096) at 68.1 vs 68.3 TFLOPS.
  - **This likely explains our "rowwise slower than FP16 at ≤224 rows"** (unverified for our shapes): check `use_fast_accum`, and check whether `torch._scaled_mm` rowwise on sm_90 picks a non-swap-AB CUTLASS kernel at small M.
- **Fast accumulation is a quality risk.** DeepSeek-V3 reports that Hopper FP8 MMA keeps only about **14 bits** of accumulation precision. They promote partial sums to FP32 every 128 elements ([summary](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/), [paper](https://arxiv.org/abs/2412.19437)). Our K is 1536, and 8960 for down-proj.
  - Mitigations: keep **down_proj in FP16** (or FP32-accum), use fast accum elsewhere, and keep the first and last 1–2 layers in FP16.
  - SmoothQuant-style migration (α≈0.5) is the next fallback.
- **Where FP8 fits:**
  - **Throughput mode (M ≈ 1.5k–7.5k rows per GEMM at B=64)** is compute-bound and FP8 dense peak is 2× FP16. Measured: Llama-3.1-8B float8 rowwise 1.28× at batch 32 (PyTorch blog). (est.) **1.3–1.6× users/s** if HR@10 holds.
  - Fuse per-token absmax and quantization into the (lazy) RMSNorm, or the previous epilogue, so activations are not re-read.
  - FBGEMM-GenAI `f8f8bf16_rowwise` has small-M tuned configs and cu126 wheels from the PyTorch index, which is an alternative to `torch._scaled_mm`.
  - **Latency mode:** FP8 at M ≤ 64 is only worth it with a swap-AB kernel. vLLM got 5–25%; low priority.
- **2:4 sparsity:** **not recommended.**
  - Sparse GEMM "falls below dense GEMM performance for small M" ([Spense, arXiv 2606.10445](https://arxiv.org/abs/2606.10445)). One user reports 2:4 Llama-2-7B decode about 12% *slower* than dense ([NVIDIA forum](https://forums.developer.nvidia.com/t/why-am-i-2-4-sparse-slower-than-dense-in-the-decode-stage-of-llama2-7b/340931)). Sparse-Marlin adds up to 65% per layer, but on Ampere and only in the memory-bound regime.
  - Quality loss on a 1.5B ranking model would need retraining.
- **FP8 KV cache:** irrelevant. KV is about 3 MB per request against 3.2 GB of weights.

---

## 6. Scoring a tree of tokens in one pass

- **Prior art** (cite these in the paper):
  - SpecInfer ([arXiv 2305.09781](https://arxiv.org/abs/2305.09781)): token-tree verification in one pass with a topology-aware mask.
  - Sequoia ([arXiv 2402.12374](https://arxiv.org/abs/2402.12374)): DP-optimal tree shape plus a **hardware-aware tree-size optimizer**, whose premise is that forward time is flat in the number of verified tokens until compute-bound.
  - DeFT ([arXiv 2404.00242](https://arxiv.org/abs/2404.00242), ICLR'25): KV-guided grouping and flattened tree KV splitting, up to 2.23× decode and 3.59× attention latency, with 73–99% less KV IO.
  - FlashInfer cascade ([blog](https://flashinfer.ai/2024/02/02/cascade-inference.html), [arXiv 2501.01005](https://arxiv.org/abs/2501.01005)): up to 31× for long shared prefixes. Their own caveat: end-to-end gains are small when shared prefixes are short.
- **Relevance to us.** The prompt KV is about 100 tokens and the tree has 88 nodes, so **attention bytes and FLOPs are negligible** next to the weights. DeFT and cascade-style KV dedup solve a problem we do not have.
- **The cost of the tree is purely the extra GEMM rows** (M 30 → 118 in layers 18–27). The useful engineering is:
  1. **Keep per-layer time flat in M** up to the H100 ridge (about M≈290 rows in FP16). Sequoia's hardware-aware sizing is the precedent for choosing 88 vs more nodes.
  2. **Put tree attention inside the fused QKV→attention→O block** (ClusterFusion-style, or as a megakernel instruction) rather than as standalone launches.
  3. **Make forward 1 → tree selection → forward 2 one graph with no host round trip.** Use fixed-size top-k and padded node counts.

---

## 7. Ranked recommendations

| # | Change | Expected gain | Evidence | Effort | Oscar H100 (CUDA 12.6, PyTorch 2.x, Triton 3.x, no root) |
|---|---|---|---|---|---|
| 1 | **Fuse each layer down to 4–5 launches**: lazy RMSNorm with γ folded into weights, QKV+bias+RoPE+KV-store epilogue, gate_up+SiLU epilogue, residual (+Σx²) epilogues on O and down, one attention launch for prompt plus tree | **−0.4 to −0.8 ms** (est. ~5 µs × ~5 boundaries × 34) | Hazy 7-instruction layer; PyTorch lazy pre-norm hides ≤90% of norm cost; DeepFusionKernel +13.2% (H100) | 30–60 h | Yes, pure Triton. Also helps A6000. |
| 2 | **PDL across the whole chain**: `config.triton.enable_pdl=True` (torch ≥2.10) plus `launch_pdl=True` and `gdc_wait`/`gdc_launch_dependents` in our Triton attention and GEMM kernels; issue first weight-tile loads before `gdc_wait`; replace cuBLAS in the latency path so it does not break the chain | **−0.15 to −0.35 ms** (est.) | vLLM #64981: −3.1% to −4.2% TPOT at batch 1 on H200 for 8B models (our kernels are shorter, so a bigger share); TRT-LLM +3% | 4–12 h | Yes (Triton ≥3.4, sm_90; graph-capturable on CUDA ≥12.3). Not on A6000. |
| 3 | **Persistent megakernel** (end state; subsumes 1, 2 and 6): static per-SM schedule per (prompt bucket, tree size), counters, smem page pool, weight prefetch | **2.9 → ~1.3–1.6 ms** (est. at 65–75% MBU) | Hazy: 78% MBU, <1 ms Llama-1B on H100, ≥1.5× SGLang; ForgeMegakernel: 74% MBU, 1.37× SGLang; Triton-dist: 1.65× over torch + CUDA graph | 150–300 h | Yes as our own CUDA (`torch.utils.cpp_extension`, `module load cuda/12.6`) or vanilla Triton with atomics. MPK builds from source without root. Hazy repo expects a cu128 torch (needs a driver ≥ R570). |
| 4 | **Per-shape GEMM dispatch** for the small GEMMs (QKV and O at all M, everything at M ≤ 32): split-K to ~1–2 waves of 132 SMs, swap-AB for M ≤ 64, epilogues in the reduce; keep cuBLAS where it wins (gate_up and down at M≥64) | **−0.1 to −0.3 ms** (est.) | SplitK on H100: +124% average for skinny shapes; vLLM swap-AB +5–25%; vLLM skinny plans −15% TPOT at M≤4 (but lose at M=8–16, so measure) | 15–30 h | Yes, Triton. |
| 5 | **Throughput: FP8 W8A8 with per-channel weights and dynamic per-token activations**, fast-accum except down_proj, first and last layers in FP16, quantization fused into the norm | **+1.3–1.6× users/s** (est.) if HR@10 stays within noise | H100 FP8 dense peak 2× FP16; rowwise+fast-accum 810 vs FP32-accum 299 TFLOPS at 2K; Llama-8B float8 rowwise 1.28× at batch 32; FP8 dynamic "effectively lossless" | 10–25 h + evaluation | Yes (`torch._scaled_mm` rowwise with `use_fast_accum`, or FBGEMM-GenAI cu126 wheels). |
| 6 | **Cross-layer and overlap hygiene**: fold the residual into the next layer's norm (no per-layer compile cut), `combo_kernels`, one graph for both forwards with no host sync, L2 prefetch (`cp.async.bulk.prefetch.L2` via `tl.inline_asm_elementwise`) of the next layer's 11 MB QKV+O weights during attention and tree kernels | **−0.05 to −0.25 ms** (est.) | Triton-dist megakernel prefetch task; TRT-LLM PDL weight prefetch; H100 L2 is 50 MB | 4–10 h | Yes. |

**Suggested order:**
1. Diagnostic (§0).
2. #2, a few hours, to measure the boundary-cost share directly.
3. #6.
4. #1 and #4 together; they share the split-K and epilogue kernel.

That should reach about 1.8–2.2 ms (est.). #3 is then the paper's "how close to the floor" experiment, and the per-SM instruction list will already exist as the Triton kernels from #1. Run #5 in parallel on the throughput track; it is independent of the latency work.

**Deprioritized, with reasons:**
- 2:4 sparsity: slower than dense at small M, and costs quality.
- W4A16: quality loss already seen, and no win at M≥64 on H100.
- FP8 KV cache: KV is negligible.
- DeFT or cascade attention: our prefix is tiny.
- Stream-K: compute-bound shapes only.
- TileRT and ETC: multi-GPU, MoE or TVM-specific.

---

## 8. Sources

- Hazy Research. [Look Ma, No Bubbles (Llama-1B megakernel)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); [TP Llama-70B megakernel](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main); [Megakernels repo](https://github.com/HazyResearch/Megakernels); [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
- [MPK (arXiv 2512.22219)](https://arxiv.org/abs/2512.22219); [mirage repo](https://github.com/mirage-project/mirage); [Event Tensor (arXiv 2604.13327)](https://arxiv.org/abs/2604.13327); [ForgeMegakernel (arXiv 2609.12379)](https://arxiv.org/abs/2609.12379); [AutoMegaKernel (arXiv 2606.09682)](https://arxiv.org/abs/2606.09682)
- [Triton-distributed megakernel docs](https://triton-distributed.readthedocs.io/en/latest/getting-started/megakernel/megakernel.html) (source in `python/triton_dist/mega_triton_kernel/`, read for the scoreboard and prefetch code); [Cohere megakernel](https://cohere.com/blog/megakernels); [TileRT](https://github.com/tile-ai/TileRT)
- [ClusterFusion (arXiv 2508.18850)](https://arxiv.org/abs/2508.18850), [code](https://github.com/xinhao-luo/ClusterFusion); [DeepFusionKernel (arXiv 2602.11808)](https://arxiv.org/abs/2602.11808); [PyTorch: Towards Free Normalization](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/); [QuACK mem-bound SOL](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)
- PDL: [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html); [Triton tutorial 11](https://github.com/triton-lang/triton/blob/main/python/tutorials/11-programmatic-dependent-launch.py) (Triton PR #6394, v3.4.0+); PyTorch Inductor `triton.enable_pdl` (PR #160928, first in v2.10.0; #174231 combo kernels); [vLLM #64981 Inductor-PDL measurements](https://github.com/vllm-project/vllm/pull/64981); [SGLang #18596 graph-capture caveat](https://github.com/sgl-project/sglang/pull/18596); [TRT-LLM B200 min-latency blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html); [NVIDIA constant-time graph launch](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements)
- GEMMs: [vLLM #64979 SM90 skinny GEMM plans](https://github.com/vllm-project/vllm/pull/64979); [vLLM #20396 FP8 swap-AB](https://github.com/vllm-project/vllm/pull/20396); [SplitK (arXiv 2402.00025)](https://arxiv.org/abs/2402.00025); [Stream-K (arXiv 2301.03598)](https://arxiv.org/abs/2301.03598)
- Quantization: [Machete](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel); [Marlin (arXiv 2408.11743)](https://arxiv.org/abs/2408.11743); [PyTorch GemLite/TorchAO/SGLang](https://pytorch.org/blog/accelerating-llm-inference/); [FP8 across accelerators (arXiv 2502.01070)](https://arxiv.org/abs/2502.01070); [Give Me BF16 or Give Me Death (arXiv 2411.02355)](https://arxiv.org/abs/2411.02355); [Qwen2.5-0.5B-FP8-dynamic](https://huggingface.co/neuralmagic/Qwen2.5-0.5B-FP8-dynamic); [Qwen2.5-1.5B w8a8](https://huggingface.co/RedHatAI/Qwen2.5-1.5B-quantized.w8a8); [DeepSeek-V3 (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) via [Colfax summary](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/); [FBGEMM-GenAI install](https://pytorch.org/FBGEMM/fbgemm_genai/development/InstallationInstructions.html); [Spense 2:4 (arXiv 2606.10445)](https://arxiv.org/abs/2606.10445); [NVIDIA forum 2:4 decode](https://forums.developer.nvidia.com/t/why-am-i-2-4-sparse-slower-than-dense-in-the-decode-stage-of-llama2-7b/340931)
- Trees: [SpecInfer (arXiv 2305.09781)](https://arxiv.org/abs/2305.09781); [Sequoia (arXiv 2402.12374)](https://arxiv.org/abs/2402.12374); [DeFT (arXiv 2404.00242)](https://arxiv.org/abs/2404.00242); [FlashInfer (arXiv 2501.01005)](https://arxiv.org/abs/2501.01005); [FlashInfer cascade blog](https://flashinfer.ai/2024/02/02/cascade-inference.html)

**Not independently verified** (seen only in search summaries): AlpinDale's Qwen3-0.6B megakernel on RTX 5090 (~1,033 tok/s); MegaQwen on RTX 3090 (527 tok/s); the tile-level SwiGLU fusion paper (arXiv 2607.02521, up to 2.47× over PyTorch on Qwen-2.5); and the 2% relative-error figure for Hopper FP8 fast accumulation.
