# Kernel techniques for RapidTinyRec decode (Qwen2-1.5B, constrained beam 50)

*Research date: 2026-11-07. Primary device: RTX A6000 (sm_86, 768 GB/s, 84 SMs, 6 MB L2, no FP8). Secondary: L40S (sm_89, 864 GB/s, FP8). Env: torch 2.14.0+cu126.*

Speedup numbers marked **(est.)** are my extrapolations from published curves. Other numbers come from the cited source. Benchmark everything on the A6000 before relying on it.

---

## 0. Budget math: where 56.5 ms can go

Weights per forward, from the Qwen2-1.5B config:

| Part | Params | BF16 bytes | Share |
|---|---|---|---|
| QKV (1536→2048) + O (1536→1536), x28 | 0.155 B | 0.31 GB | 10% |
| gate+up (1536→17920) + down (8960→1536), x28 | 1.156 B | 2.31 GB | 75% |
| LM head (tied, 152,265×1536) | 0.234 B | 0.47 GB | 15% |
| **Total** | **1.544 B** | **3.09 GB** | |

- **BF16:** 3.09 GB at about 690 GB/s achievable is roughly 4.5 ms per forward. Four forwards come to about 18 ms. Kernel-boundary overhead adds about 1–2 µs per kernel. At roughly 300 kernels per forward that is another 1–2.5 ms per request, so a realistic BF16 target is **about 20–22 ms**.
- **W8A16** (int8 or fp8 weights, LM head int8): about 1.55 GB per forward, or 2.3 ms. Expect **about 11–13 ms** per request.
- **W4A16** (g128) plus an int8 LM head: about 0.93 GB per forward, or 1.35 ms. Expect **about 7–9 ms** per request. Kernel fixed costs dominate at this point.
- **Arithmetic intensity:** M=50 gives about 50 FLOP/byte. The A6000 ridge point is about 200 (155 dense BF16 TFLOPS / 768 GB/s). Decode is therefore firmly memory-bound up to M≈150–200. At M=800–1600 (B=16–32) the GEMMs become compute-bound: about 4.9 TFLOP per forward at B=32, which is ≥35 ms per forward in BF16. Different tricks apply in that regime.

**Implication.** About 35 ms of the current 56.5 ms is *not* weight streaming. The likely causes are HF beam-search bookkeeping, host syncs, KV expanded 50× and reordered every step, the full-vocab log-softmax and top-k, and kernel count. Fix those structural costs before tuning GEMMs.

---

## 1. What "Meta kernel challenge" most plausibly refers to

### Most likely: the Muse Spark Open Kernel Challenge (OKC 2026)

- **Organizers and timing:** Run by **Paradigm with Meta and Andromeda**. Kernel track ran Oct 22 – Nov 4, 2026. A serving track (patching SGLang on 8×H200) runs through Nov 19. Prize is a $100k share plus a signed H200. Submissions are **Apache-2.0**, so the code is copyable.
- **Scoring:** Speed up layers of Meta's Muse Spark Open (816B NVFP4 MoE) against SGLang on one H200. Score is the geometric-mean speedup over shapes. The judge times under **CUDA-graph replay with cold L2**, with no lower precision than SGLang, a fresh input per call, and an untimed `prepare()`.
- **Problems:** `muse_bf16_proj` (skinny BF16 GEMV/GEMM, M = 1, 16, 32 decode and 8, 128, 256 verify), `muse_attention_decode` (FP8 KV split-KV decode), `muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer`. This maps almost one-to-one onto our problem: skinny BF16 projections at M≈50, weight-only quantization, and a fused decode layer.
- **Links:** [Paradigm announcement](https://www.paradigm.xyz/writing/muse-spark-open-kernel-challenge), [openkernelchallenge.dev](https://openkernelchallenge.dev/).
- **Public write-ups:** [trxphxle](https://github.com/trxphxle/okc-2026-muse-kernels) (decode layer 2.18×), [Mithun-033 bf16_proj journey](https://github.com/Mithun-033/Muse-Spark-Open-Kernel-Challenge), [georgxzh](https://github.com/georgxzh/muse-kernel-challenge), [infraax](https://github.com/infraax/okc-kernel-challenge), [yesthatwilly](https://github.com/yesthatwilly/open-kernel-challenge-2026), [vjeux](https://github.com/vjeux/okc-2026). Final winners were not yet published when I checked.

**Tricks that transfer to Ampere** (Hopper-only features such as TMA, WGMMA, PDL and NVFP4 do not):

1. **Per-shape backend dispatch, chosen offline.** Use a custom split-K kernel for small M and cuBLAS for larger M, selected in `prepare()` by timing each shape. A uniform Triton GEMM for all shapes *regressed* to 0.69×, because there were too few CTAs at small M (Mithun-033).
2. **Two-phase, non-atomic split-K.** Write FP32 partials to a scratch buffer, then run a vectorized reduce. `tl.atomic_add` split-K serialized on L2 atomics: 13.5 µs became 18.2 µs.
   - The alternative is a single-kernel split-K with a GPU-scope acquire-release "ticket": the last CTA sums partials in fixed order, which keeps the result deterministic, then resets the counter (georgxzh).
   - Split-K won only at very small token counts. At T=16–32, cuBLAS was equal or better.
3. **Size grids to whole waves**, for example multiples of the SM count (84 on A6000).
4. **Streaming GEMV for M≤8:** 16-byte `ld.global.nc.v4` loads with FP32 FMA and warp-shuffle reduce reached about 98% of HBM bandwidth. This is relevant only if a step ever has tiny M.
5. **Lossless BF16 weight compression ("C12").** 16 bytes of BF16 are packed into 12 using exponent windowing plus an outlier table, decoded with `lop3`/`prmt`. That is about 25% less DRAM traffic with bit-exact output. It is the only "free" byte reduction if outputs must match BF16 exactly. The peer-reviewed analogue is ZipServ (§2, T9).
6. **Fusing at kernel boundaries** (QK-norm + RoPE + cache write, residual + RMSNorm) helped. Some fusions *lost* 2–4% because they removed overlap between kernels (trxphxle). Measure every fusion.
7. **Marlin was very hard to beat for W4A16.** Custom W4A16 GEMMs reached only 0.58–0.67× of Marlin. A tensor-core "GEMV" for BF16 at M≤16 also lost to tiled split-K (georgxzh, infraax).
8. **Methodology:**
   - Time candidates under CUDA-graph replay; eager timing misranked kernels.
   - Run-to-run noise is ±1.5–3%, so use paired A/B for changes under 2%.
   - At small M the floor is DRAM bandwidth, and removing launches alone can't close the gap to it.

### Other candidates (less likely, still useful)

| Candidate | What | Transferable lesson |
|---|---|---|
| [GPU MODE NVFP4 hackathon](https://luma.com/9n27uem4) (Nov 2025–Feb 2026, B200; batched GEMV, GEMM, gated dual GEMM, grouped GEMM) | KernelBot leaderboards | GEMV is memory-bound, so use 128/256-bit vector loads, `L1::no_allocate` for weights and `evict_last` for reused activations, compile-time K specialization, and low register counts. Double-buffered cp.async did *not* help GEMV ([summary](https://slop.sus.cat/tools/hackathons-2026/webapp/reports/past-winners-gpu-mode.html)). |
| AMD × GPU MODE E2E speedrun (MI355X, Mar–May 2026; MXFP4 GEMM, MLA decode, MXFP4 MoE) | $1.1M | At small M keep activations in BF16 and use weight-only kernels. The common win was removing launches, copies and syncs. |
| [PyTorch Helion hackathon](https://pytorch.org/event/helion-hackathon/) (Mar 14, 2026) and Helion DSL (Meta/PyTorch) | Autotuned Triton-generating DSL | Helion autotuning is an option for writing split-K and fused kernels quickly. |
| Meta KernelEvolve ([blog](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/)), BackendBench, KernelLLM, KernelBench | Agentic or LLM kernel generation for Meta ads/recsys | Search-based autotuning, with correctness gates against a reference. |
| Meta [generative-recommenders](https://github.com/meta-recsys/generative-recommenders) (HSTU, M-FALCON) | Recsys kernels | M-FALCON amortizes one user-context pass across many candidates, the same idea as a shared-prefix beam. |
| GR-specific: [NVIDIA sid-gr-inference](https://github.com/NVIDIA/recsys-examples/tree/main/examples/sid-gr-inference) ([blog](https://developer.nvidia.com/blog/how-generative-recommenders-are-redefining-recsys-at-scale/)), [GRACE](https://arxiv.org/html/2608.00938) | Semantic-ID beam-search engines | See below. |

**GR-specific details.**
- **NVIDIA sid-gr-inference** separates KV into shared ContextKV, per-beam BeamKV, and BeamPath ancestry. It gets 2.1–2.3× lower latency than SGLang beam search: Qwen3-1.7B, beam 256, 3 SID tokens, H100, 47.7 ms at batch 4 with ctx 1000.
- **GRACE** stores the trie on GPU in CSR, carries a node-id per beam, and fuses the constraint mask into the top-k path. It captures the whole fixed-length decode loop in one CUDA graph and got 11× on its decoder.

---

## 2. Ranked techniques, by expected latency win for our workload

### T1. One CUDA graph per request, GPU-resident constrained beam search, shared-prefix KV (largest win: 56.5 → ~22–26 ms BF16, est.)

Replace HF `generate` with a hand-written Qwen2 forward plus vectorized beam search. The request is prefill (padded to a bucket), top-50 from the last prompt position, then 2–3 decode steps.

- **Prefix KV:** Keep **one** copy of the prompt KV, shaped `[L, 2, Hkv=2, P, 128]`. Do not expand it 50× and do not `reorder_cache` it.
- **Generated-token KV:** Store the ≤3 generated tokens per beam in `[B=50, L, 2, Hkv, T≤4, 128]`, about 2.9 MB in total. Reorder with a single `index_select(parent)` per step across all layers, or keep a BeamPath ancestry table and gather inside attention.
- **Trie on GPU:** Store a 256-bit child bitmask per trie node (`[num_nodes, 8] int32`, 32 B per node, about 34 MB for 1M items with 4 levels) plus `child_base[node]`.
  - The next node is `child_base[node] + rank`, where `rank = mask_bool.cumsum(-1) - 1` gathered at the chosen code.
  - Everything is fixed-shape and graph-capturable: no `nonzero`, no `.item()`.
- **Beam selection:** `cand = beam_score[:,None] + logit_allowed - lse[:,None]`. Mask invalid entries to -inf, then `topk(cand.view(-1), 50)`. The parent is `idx // C` and the code is `idx % C`.

Capture the whole request (prefill plus all steps) in **one** `torch.cuda.CUDAGraph`. Use one graph per prompt bucket (for example 96 and 128); padding costs almost nothing because M≤128 is memory-bound. See §4 for the recipe.

### T2. Cheap structural fixes (each about 0.3–2 ms, est.)

- **Concatenate weights.** Concatenate `q/k/v` into one [2048,1536] weight with bias, and `gate/up` into one [17920,1536] weight. That takes 7 GEMMs per layer down to 4, with bigger, better-utilized GEMMs.
- **Prefill LM head on the last position only.** Slice `h[:, -1]` before the LM head (HF's `logits_to_keep=1`).
- **Restrict logits and use exact LSE.**
  - If SID codes for each level are a contiguous token range, compute `h @ W[range].T` only for the 256 allowed codes, plus the full-vocab LSE (needed to match HF exactly, because HF applies `prefix_allowed_tokens_fn` after `log_softmax`).
  - This avoids writing and reading 50×152k logits and running top-k over 7.6M entries.
  - If you accept renormalizing within allowed tokens, you can skip the 0.47 GB LM-head read. That changes scores; validate recall@K first.
- **Check ECC and the GPU model.** A6000 with ECC on loses roughly 10–15% bandwidth (`nvidia-smi -q -d ECC`). For memory-bound M=50, an **RTX 3090 (936 GB/s)** can beat the A6000 by up to 1.2×. The L40S (864 GB/s plus FP8) is the fastest card available.

### T3. Weight-only W8A16 (near-lossless; about 1.6–1.8× on GEMM time → ~12–14 ms total, est.)

Best accuracy/latency trade-off for a ranking model. Options, in order:

1. **Marlin 8-bit (`uint8b128`) or FP8-Marlin, from vLLM.**
   - sm_80+ (A6000 included); BF16 activations; group size −1 or 128.
   - Shape rules: N%64==0 and K%128==0. All Qwen2 projections pass (K = 1536 or 8960). The LM head must be padded to 152,320.
   - See T4 for how to get the kernel.
2. **torchao `Int8WeightOnlyConfig`.** Only fast under `torch.compile(mode="max-autotune-no-cudagraphs")`. Inductor's `prologue_fusion` (default on) fuses the int8→bf16 cast into the Triton mm template.
   - Without compile it *materializes* dequantized weights and is slower than BF16. torchao's own vLLM table shows int8-weight-only decode at 0.65× of BF16 on H100.
3. **GemLite `A16W8_INT8`** (pure Triton, so it runs on any torch and Triton version). See §3.

### T4. W4A16 with Marlin (about 2.5–3× on GEMMs at M=50, so roughly 8–10 ms per request with fixed costs, est.; ranking risk)

- **Kernel behaviour:** Marlin is near-ideal (about 3.9×) up to M≈16–32 and still clearly faster at 64–128 on large matrices ([paper](https://arxiv.org/abs/2408.11743), A10).
  - Our K=1536 layers are small, so expect less on QKV/O and close to the paper's figure on gate_up and down.
  - Avoid at M≥256 (batched mode).
- **Use vLLM's `gptq_marlin`, not the original IST-DASLab repo.** The original is FP16-only, 4-bit, g128 or per-channel. vLLM's version handles BF16, 4- and 8-bit, and has minimum capability 80.
- **Ways to get it with torch 2.14:**
  - (a) `pip install -v gptqmodel` (bundles Marlin; JIT-compiles with ninja, so run `module load cuda/12.6` for nvcc). Use its Marlin QuantLinear and repack APIs.
  - (b) A separate env pinned to torch 2.8+cu126 with `pip install kernels`, then `get_kernel("kernels-community/quantization")`. Builds stop at **torch 2.8**, so this does not work on torch 2.14.
  - (c) A separate vLLM env, calling `vllm._custom_ops.gptq_marlin_gemm` after `gptq_marlin_repack` and `marlin_permute_scales`.
- **Quantizer:** Use GPTQ (via gptqmodel) or HQQ (`pip install hqq`, no calibration data) at g128. Gate on top-50 overlap or recall@K against BF16, not perplexity.

### T5. Skinny-GEMM tuning plus epilogue fusion (about 1–3 ms, est.)

**Grid sizes.** At M=50 with 128-wide N tiles, QKV (N=2048) launches 16 CTAs, and O and down (N=1536) launch 12 CTAs on 84 SMs. That cannot saturate DRAM unless cuBLAS picks split-K.

**Measure first.** Use `torch.profiler` and compute achieved GB/s per GEMM as weight bytes over kernel time. If any GEMM is below about 600 GB/s, write a Triton split-K GEMM:
- `BLOCK_M=64` covers all 50 rows; `BLOCK_N` is 32–64.
- Choose `SPLIT_K` so the CTA count is about 2–4× 84.
- Write FP32 partials, then reduce. Keep partial traffic small relative to weights (for QKV, `SPLIT_K≤4`).

**Put epilogues in the reduce kernel:**
- QKV: + bias, RoPE, KV-cache write.
- gate_up (interleave gate and up rows): `silu(g)*u`.
- down / O: + residual, then the *next* RMSNorm, since one CTA per row holds all 1536 columns.

That is about 8–9 kernels per layer, each near the bandwidth roofline. Starting points: GemLite `gemm_splitK` kernels (Apache-2.0) and the OKC bf16_proj submissions (Apache-2.0).

**Inductor settings:**
- `max_autotune` can choose Triton templates with small `BLOCK_N` and fuse epilogues.
- `coordinate_descent_tuning=True` tunes the pointwise and reduction kernels.
- Decompose-K exists in recent Inductor but targets K≫N; it won't fire for our shapes ([blog](https://shreyansh26.github.io/post/2026-06-21_decompose-k-triton/)).

### T6. LM head (about 0.3 ms per forward at M=50; several ms at B=32)

- **Quantized copy.** Keep the BF16 embedding for lookups and add a **separate int8 or FP8 per-row copy** for the LM-head GEMM, so the 0.47 GB read becomes 0.23 GB.
- **Fused LM-head + LSE kernel.** Use a Triton kernel with a grid over vocab tiles (152,320/128 = 1190 CTAs).
  - Each CTA computes its [64×128] logits tile and writes per-row (max, sumexp) partials (about 0.6 MB).
  - CTAs overlapping the allowed SID range write their logits.
  - A tiny merge kernel then reduces partials with flash-decoding LSE merge math.
  - At B=32 this avoids a 973 MB FP32 logits tensor.
  - Reference: Apple Cut-Cross-Entropy `cut_cross_entropy/cce_lse_forward.py` (Triton, Ampere+). Its license is Apple's custom permissive grant (use, modify and redistribute; keep the notice).
  - Quick hack: `linear_cross_entropy(e, c, targets, reduction="none", return_lse=True)`. Check that it is graph-capturable.
  - Liger's FusedLinearCrossEntropy (BSD-2) chunks over tokens and still materializes chunk logits, so it is less suited.

### T7. Batched mode (M = 50×B, up to 1600)

- **Up to M≈150–200 (B≤3–4) on A6000 BF16,** batching is nearly free per request, because weights are amortized.
- **Above that, GEMMs are compute-bound,** so W4A16 stops helping.
  - **Ampere:** Use W8A8 INT8. A6000 dense INT8 is about 310 TOPS against about 155 TFLOPS for BF16.
    - Call `torch._int_mm(x_i8, w_i8.t())`. Historically it required M>16 and K,N multiples of 8; pad if needed.
    - Or use torchao `Int8DynamicActivationInt8WeightConfig()` with `torch._inductor.config.force_fuse_int_mm_with_mul=True` so the scale epilogue fuses.
    - Fuse per-token activation quantization into the preceding RMSNorm.
    - Expect about 1.6–1.9× at M≥800 (est.). Check accuracy; SmoothQuant may be needed.
  - **L40S:** Use FP8 rowwise W8A8, about 733 dense FP8 TFLOPS against 362 BF16 (§3).
- **Pack prefills with different prompt lengths (varlen)** rather than padding. Optionally piggyback new users' prefill rows onto others' decode steps (chunked-prefill style) to save a full weight pass per request.

### T8. Long shots (high effort)

- **Lossless compression fused into GEMM.** [ZipServ](https://arxiv.org/html/2603.17435v1) (ASPLOS'26, [code](https://github.com/HPMLL/ZipServ_ASPLOS26)) uses a fixed 3-bit exponent code. Weights shrink to about 71%, bit-exact. Kernel speedups average 1.31–1.36× over cuBLAS on RTX 4090 and L40S. It was tested on Ada, not Ampere. This is the OKC "C12" idea in published form.
- **Megakernel.** [Mirage MPK](https://github.com/mirage-project/mirage) claims A100 support and 1.0–1.7× over SGLang/vLLM. HazyResearch Megakernels is H100/B200 only. A big lift for our custom beam logic.

---

## 3. Weight-quantization kernel matrix (sm_86, torch 2.14 cu126)

Speedups are kernel-level against BF16 cuBLAS on A6000-class GPUs and are rough **(est.)**. Shapes are Qwen2-1.5B, which are smaller than the papers' shapes.

| Kernel | Runs on sm_86? | Install | M=16 | M=50 | M=128 | M=800 | nn.Linear API |
|---|---|---|---|---|---|---|---|
| **Marlin W4A16 (vLLM gptq_marlin)** | Yes (cap ≥80) | `gptqmodel` (JIT, needs nvcc 12.6), or a vLLM env, or torch-2.8 env + `kernels` | ~3× | ~2.5× | ~1.5–2× | ≤1× | gptqmodel Marlin QuantLinear, or `gptq_marlin_gemm(a, None, b_q, b_scales, ..., workspace, wtype, M, N, K)` after `gptq_marlin_repack` and `marlin_permute_scales` |
| Marlin W8A16 / FP8-Marlin | Yes | same | ~1.8× | ~1.7× | ~1.3× | ~1× | same, `uint8b128` or fp8 |
| torchao Int4 (tinygemm, `tile_packed_to_4d`) | Yes (ATen `_weight_int4pack_mm`) | `pip install torchao` (≥0.17 is ABI-stable for torch ≥2.11) | ~2.5× | ~1.2–1.5× | ~1× | <1× | `quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq"), filter_fn=...)` |
| torchao Int8 weight-only + compile | Yes | torchao | ~1.7× | ~1.5× | ~1.1× | ~1× | `Int8WeightOnlyConfig()`; requires `torch.compile` max-autotune |
| **GemLite** A16W8 / A16W4 (Triton) | Yes (Triton, sm_80+) | `pip install gemlite` or git main | ~2–3× | ~1.5–2.5× | ~1.2× | ~1× | `A16W8_INT8().from_linear(lin)`, `A16W4_HQQ_INT`, `GemLiteLinear(...).pack(...)`, `patch_model(...)`. Cache autotune per GPU with `gemlite.cache_config/load_config` |
| W8A8 INT8 (`torch._int_mm`) | Yes (IMMA) | core | n/a | ~1.5× | ~1.6× | ~1.7–1.9× | torchao `Int8DynamicActivationInt8WeightConfig()` |
| FP8 W8A8 (`torch._scaled_mm`) | **No**; L40S only | core | — | ~1.6× | — | ~1.8× | torchao `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())` (needs cap ≥8.9) |
| Machete | **No** (Hopper sm_90, TMA/WGMMA) | — | | | | | |
| FLUTE | Runs on A6000, but kernels are shape-specialized and tuned for Llama shapes in fp16 | `flute-kernel` | Skip | | | | |
| bitsandbytes NF4 / LLM.int8 | Runs | `bitsandbytes` | <1× at M>1 (dequantize then GEMM) | | | | Skip |
| AWQ (AutoAWQ, archived) | Runs | — | Slower beyond bs≈8 | | | | Repack AWQ weights to Marlin instead |
| exllamav2/v3, QServe (W4A8KV4) | sm_80+ | Engine-centric | Awkward as a raw nn.Linear | | | | Skip |

### FP8 on L40S with `torch._scaled_mm`

```python
W8 = (W.float() / (ws := W.abs().amax(1, keepdim=True).float().clamp_min(1e-12) / 448)).to(torch.float8_e4m3fn)  # [N,K] row-major
xs = x.abs().amax(1, keepdim=True).float().clamp_min(1e-12) / 448                                              # [M,1]
y = torch._scaled_mm((x / xs).to(torch.float8_e4m3fn), W8.t(),   # mat_b must be column-major -> W8.t()
                      scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)  # rowwise: (M,1) and (1,N) fp32
```

- The public API in 2.14 is `torch.nn.functional.scaled_mm(mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b, ..., output_dtype=bf16)` ([docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_mm.html)).
- **Dimensions:** Pad M, K and N to multiples of 16 (50→64; LM head 152,265→152,320, masking the pad logits to −inf).
- **Rowwise on sm_89** uses a CUTLASS kernel added in [PR #144348](https://github.com/pytorch/pytorch/pull/144348) (Jan 2025). Older builds failed rowwise on Ada ([#130359](https://github.com/pytorch/pytorch/issues/130359)), so smoke-test on the L40S. Tensorwise uses cuBLASLt and is safe.
- `use_fast_accum` is documented as Hopper-only.
- Fuse activation absmax and quantization into RMSNorm via Inductor.

---

## 4. CUDA graphs + torch.compile recipe

```python
import torch, torch._inductor.config as ic, torch._dynamo.config as dc
ic.coordinate_descent_tuning = True        # tunes pointwise/reduction kernels (RMSNorm, softmax)
dc.automatic_dynamic_shapes = False        # never silently go symbolic on the 2nd prompt length
dc.recompile_limit = 64                    # (older name: cache_size_limit)
# export TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR to persistent scratch, one dir per GPU model

prefill_c = torch.compile(prefill_impl, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
decode_c  = torch.compile(decode_impl,  mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)

@torch.inference_mode()
def run_request(S):                 # S: preallocated static tensors (ids, prompt_len, kv, beams, trie...)
    prefill_c(S)                    # writes prefix KV, step-0 top-50 from last position
    for t in range(1, N_LEVELS):    # unrolled at capture; t is a python const -> <=4 specializations
        decode_c(S, t)

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s), torch.inference_mode():
    for _ in range(3): run_request(S)          # compile + autotune happen here, NOT during capture
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):                      # pool=g_other.pool() to share memory across buckets
    run_request(S)

def serve(ids):                                # per request: only copy_/fill_ into static buffers
    S.ids.zero_(); S.ids[0, :len(ids)].copy_(ids); S.prompt_len.fill_(len(ids))
    g.replay(); return S.out_tokens.cpu(), S.out_scores.cpu()   # the single host sync
```

**Pitfalls:**
- Use `mode="reduce-overhead"` or `"max-autotune"` (which use cudagraph trees) **or** manual capture, never both; nesting causes capture errors. SGLang's graph runner uses the same pattern: compile with `max-autotune-no-cudagraphs`, then capture its own graphs.
- Any `.item()`, `.cpu()`, `nonzero`, boolean-mask indexing, Python branching on tensor values, or data-dependent shapes either breaks capture or silently bakes in capture-time values. Use `torch.where`, fixed-size `topk`, and `index_copy_`/`scatter_` with tensor indices for KV writes at position `prompt_len + t`.
- Dynamo guards are checked **only at capture**. Replay ignores any Python-side change (new trie tensor object, new dtype). Mutate static tensors in place.
- Autotune or recompiles triggered inside capture give "operation not permitted when stream is capturing". Warm up every bucket and step first.
- Create static buffers and run warmup, capture and replay all under the same `inference_mode`/`no_grad` regime.
- `torch.topk` tie-breaking differs from HF's path. Validate with top-K overlap, not exact equality.
- `freezing=True` lets Inductor constant-fold weight layouts. Optional, and requires `no_grad`.
- `combo_kernels=True` (experimental) horizontally fuses small independent kernels. Try it to cut kernel count.
- Use `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()` to skip recompiles on fresh SLURM nodes.
- On Ampere there is no PDL, so every kernel boundary costs a fill and drain. Kernel count is a first-class metric (`nsys` or a profiler kernel count per forward).
- If compute nodes lack internet, run `pip download` wheels on the login node first.

---

## 5. Shared-prefix beam attention

KV is tiny: 100 tokens × 2 KV heads × 128 × 2 tensors × 2 bytes ≈ 102 KB per layer. Skip cascade libraries. Fold all 50 beams × 6 query heads per KV head into one GEMM against the shared prefix, and concatenate scores with the private suffix, so a single softmax needs no LSE merge.

```python
def beam_attn(q, kp, vp, pmask, ks, vs, smask, scale):
    # q [B,12,D]; kp/vp [2,P,D] shared; ks/vs [B,2,T,D] private (T<=4); pmask [P], smask [B?,T] additive
    B, Hq, D = q.shape; Hkv = kp.shape[0]; qg = q.view(B, Hkv, Hq // Hkv, D)
    sp = torch.einsum('bhgd,hpd->bhgp', qg, kp).float() * scale + pmask
    ss = torch.einsum('bhgd,bhtd->bhgt', qg, ks).float() * scale + smask
    p = torch.cat([sp, ss], -1).softmax(-1).to(q.dtype); P = kp.shape[1]
    o = torch.einsum('bhgp,hpd->bhgd', p[..., :P], vp) + torch.einsum('bhgt,bhtd->bhgd', p[..., P:], vs)
    return o.reshape(B, Hq, D)
```

- **Prefill (1 row, ~100 tokens):** `F.scaled_dot_product_attention(is_causal=True)`. PyTorch's built-in FA2 backend supports sm_86.
- **FlashInfer.** It provides `merge_state(v_a, s_a, v_b, s_b)` and `MultiLevelCascadeAttentionWrapper`. Current 0.7.x lists CUDA 12.9 / 13.0 prerequisites ([install](https://docs.flashinfer.ai/installation.html)), so it is not a clean install on torch cu126. It would need a cu128+ env. Not worth it at this KV size.
- **flash-attn 2.** Supports sm_86, but prebuilt wheels lag torch versions, so torch 2.14 means a long source build. FA3 is Hopper-only. Not needed.
- **FlexAttention.** Works on Ampere and can express the prefix + suffix mask. Also unnecessary here.

---

## 6. Fused kernels worth writing or copying

| Kernel | Do it? | Copy from (license) |
|---|---|---|
| Residual + RMSNorm | Let Inductor fuse it. Better still, put it in the split-K reduce epilogue of O and down. | Liger `rms_norm.py` (BSD-2), FlashInfer `fused_add_rmsnorm` (Apache-2.0), vLLM (Apache-2.0) |
| SwiGLU | Fuse into the gate_up reduce epilogue with interleaved weights | Liger `swiglu.py` (BSD-2) |
| QKV bias + RoPE + KV write | One Triton kernel, or Inductor | vLLM rotary (Apache-2.0), Liger `rope.py` (BSD-2) |
| Split-K skinny GEMM with epilogues | Yes (T5) | GemLite `gemm_splitK` (Apache-2.0); OKC bf16_proj submissions (Apache-2.0 per contest rules) |
| LM head GEMM + online LSE + allowed-logit gather | Yes (T6) | Apple CCE `cce_lse_forward.py` (Apple permissive license) |
| LSE-normalize + beam-score add + trie mask + top-k over 50×256 | Plain torch ops plus `torch.topk` under compile; the data is tiny (12.8k elements) | GRACE fuses this in CUDA, but it is not needed at our size |

---

## 7. Measurement protocol (borrowed from OKC judges)

- **Timing:** Time only `g.replay()`, with `torch.cuda.Event`. Report the median of at least 200 replays.
- **A/B comparisons:** Interleave A and B in paired runs. Expect ±2% noise. Run on an exclusive node and record clocks and ECC state.
- **Per-kernel bandwidth table:** `bytes_moved / time` for each GEMM, against about 690 GB/s achievable on A6000.
- **Correctness gate per change:** Run a held-out set of 500+ requests. Check exact top-1, top-50 set overlap ≥ X, and recall@K against the HF BF16 reference. Quantized variants must pass this gate, mirroring OKC's "no lower precision than baseline" rule.

---

## Key pitfalls (summary)

- **Wrong order of work.** Quantizing before fixing structure: about 35 ms of the 56.5 ms is not GEMM.
- **Quantization without fused kernels.** torchao int8 or int4 without `torch.compile`, or bitsandbytes, is *slower* than BF16 at M=50.
- **W4A16 in batched mode.** At M≥256 it loses to BF16. Use W8A8 INT8 (Ampere) or FP8 (L40S).
- **LM head.** Vocab 152,265 is not a multiple of 64/128/256. Pad it, and mask the pad logits to −inf before the LSE.
- **Exact HF parity** needs a full-vocab LSE. Renormalizing within allowed tokens changes beam ranking.
- **Hopper-only material.** Machete, FA3, PDL, TMA/WGMMA tricks from the OKC entries, and HazyResearch megakernels do not apply to A6000 or L40S.
- **Prebuilt binaries lag torch 2.14.** HF `kernels-community/quantization` builds stop at torch 2.8; FlashInfer 0.7.x lists CUDA 12.9/13.0 prerequisites; flash-attn wheels lag. Prefer pure-Triton paths (GemLite, custom Triton, torchao) or JIT builds with `module load cuda/12.6`.
- **Separate caches per GPU model.** Autotune results for A6000 and L40S must live in separate Triton/Inductor/GemLite cache files.
