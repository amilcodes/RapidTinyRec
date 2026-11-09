#!/bin/bash
# End-to-end fusion: SASRec computed inside the CUDA graph (side stream) vs precomputed bonus rows, same allocation.
# usage: collab_job.sh office|industrial
S=/oscar/scratch/aagraw73/speedrun/runs
COMMON="--engine exacttrie --norm cbtrie --weights 1,0.75,0 --attn triton --fuse-first --template-cache --tree-fast \
 --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --bonus-beta 0.5"
if [ "$1" = industrial ]; then
  export SPEEDRUN_DATASET=industrial REL=/oscar/scratch/aagraw73/speedrun/release_ind
  BR="--late-entry-a 18:$S/casInd/out/bridge_ind/bridge_b_e18_w.pt --late-entry 18:$S/casInd/out/bridge_ind/bridge_e18_cas.pt"
  SAS=$S/fuseI/out/sasrec
else
  BR="--late-entry-a 18:$S/bridgeBW/out/bridge/bridge_b_e18_w.pt --late-entry 26:$S/bridgeC26/out/bridge/bridge_e26_cas.pt"
  SAS=$S/fuseC/out/sasrec
fi
bash scripts/run_cfgs.sh \
 C0_file "$COMMON $BR --n-prefix 16 --n-expand 50 --bonus-file $SAS/sasrec_test.pt --modes latency,quality --repeats 2" \
 C1_graph "$COMMON $BR --n-prefix 16 --n-expand 50 --collab $SAS --modes latency,quality --repeats 2" \
 C2_k500_graph "$COMMON $BR --n-prefix 256 --n-expand 500 --ks 100,200,500 --collab $SAS --modes latency --repeats 1" \
 C2b_k500_M512_graph "$COMMON $BR --n-prefix 512 --n-expand 500 --ks 100,200,500 --collab $SAS --modes latency --repeats 1" \
 C3_tp8_graph "$COMMON $BR --n-prefix 16 --n-expand 50 --collab $SAS --quant fp8 --modes throughput --throughput-batches 64,128" \
 $EXTRA
