#!/bin/bash
# Same-GPU comparison against FlashRec (public SID-GR engine) on our exact workload, one allocation.
# usage: flashrec_job.sh smoke|latency|throughput
#   smoke       FlashRec fp16 on 64 rows: does the patched Qwen2 path load, capture and return valid SIDs
#   latency     FlashRec fp16 / bf16 / fp8 (batch-1 latency, all-user quality); ours: beam-50 under FlashRec's
#               objective (agreement), beam-50 MiniOneRec semantics, certified + bridges, + SASRec in the graph
#   throughput  FlashRec fp16 / fp8 at B = 16..128; ours: certified + bridges, FP16 and dynamic FP8
S=/oscar/scratch/aagraw73/speedrun
FP=$S/envs/flashrec/bin/python
W=$OUT/workload
$PY scripts/export_workload.py $W 2>&1 | tail -2
CK=$REL/Office_ckpt
COMMON="--engine exacttrie --norm cbtrie --weights 1,0.75,0 --attn triton --fuse-first --template-cache --tree-fast \
 --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --n-prefix 16 --n-expand 50"
BR="--late-entry-a 18:$S/runs/bridgeBW/out/bridge/bridge_b_e18_w.pt --late-entry 26:$S/runs/bridgeC26/out/bridge/bridge_e26_cas.pt"
SAS=$S/runs/fuseC/out/sasrec
# FlashRec JIT-compiles its trie kernel (tvm-ffi) and FlashInfer its attention kernels: both need nvcc, otherwise
# FlashRec falls back to PyTorch paths. Its own venv only (our PYTHONPATH would shadow its packages).
CUDA_MOD=${CUDA_MOD:-cuda/12.9.0-cinr}
module load $CUDA_MOD 2>/dev/null || echo "module load $CUDA_MOD failed"
export PATH=$S/envs/flashrec/bin:$PATH      # ninja (JIT builds) lives in FlashRec's venv
echo "nvcc: $(command -v nvcc || echo none)  ninja: $(command -v ninja || echo none)  CUDA_HOME=${CUDA_HOME:-}"
fr() { echo "=== FLASHREC $1  $(date -u +%T)"; shift
  PYTHONPATH= $FP scripts/flashrec_bench.py --model $CK --workload $W "$@" 2>&1 \
   | grep --line-buffered -E '^\{"stage"|Error|Traceback|error:|JIT|FlashRec ready|radix warmup|fallback|trie' | cut -c1-3000; }
case $1 in
smoke)
  fr F_smoke --out $OUT/F_smoke --precision fp16 --modes latency --repeats 1 --batches 16
  ;;
latency)
  fr F_fp16 --out $OUT/F_fp16 --precision fp16 --modes latency,quality --batches 16
  fr F_bf16 --out $OUT/F_bf16 --precision bf16 --modes latency,quality --batches 16
  fr F_fp8 --out $OUT/F_fp8 --precision fp8 --modes latency,quality --batches 16
  bash scripts/run_cfgs.sh \
   O1_beam_cb "--norm codebook --suffix none --fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency,quality --repeats 2" \
   O0_beam "--fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency --repeats 2" \
   O2_cert "$COMMON $BR --modes latency --repeats 2" \
   O3_cert_fus "$COMMON $BR --collab $SAS --bonus-beta 0.5 --modes latency --repeats 2"
  ;;
cand)
  # FlashRec's per-beam pre-cut (top 2 x beam of the codebook before the trie mask) vs the full codebook (exact beam
  # under its objective), same allocation; and its GPU kernel time per request
  fr F_fp16_prof --out $OUT/F_fp16_prof --precision fp16 --modes none --profile 50 --batches 16
  fr F_fp16 --out $OUT/F_fp16 --precision fp16 --modes latency --batches 16
  fr F_fp16_c256 --out $OUT/F_fp16_c256 --precision fp16 --candidates 256 --modes latency,quality --batches 16
  ;;
throughput)
  fr F_tp16 --out $OUT/F_tp16 --precision fp16 --modes throughput --batches 16,32,64,128
  fr F_tp8 --out $OUT/F_tp8 --precision fp8 --modes throughput --batches 16,32,64,128
  bash scripts/run_cfgs.sh \
   T0_beam "--fuse-first --template-cache --dtype float16 --attn triton --compile default --modes throughput --throughput-batches 16,32,64" \
   T1_cert "$COMMON $BR --modes throughput --throughput-batches 16,32,64,128" \
   T2_cert_fp8 "$COMMON $BR --quant fp8 --modes throughput --throughput-batches 16,32,64,128"
  ;;
esac
