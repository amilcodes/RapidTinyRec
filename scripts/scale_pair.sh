#!/bin/bash
# Same-allocation latency: real catalog vs synthetic 1M-SID catalog (catalogs from runs/scaleA).
S=/oscar/scratch/aagraw73/speedrun/runs
BR="--late-entry-a 18:$S/bridgeBW/out/bridge/bridge_b_e18_w.pt --late-entry 26:$S/bridgeC26/out/bridge/bridge_e26_cas.pt"
COMMON="--engine exacttrie --norm cbtrie --weights 1,0.75,0 --attn triton --fuse-first --template-cache --tree-fast \
 --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 $BR --modes latency --repeats 2 --n-expand 50"
C=$S/scaleA/out/catalogs
bash scripts/run_cfgs.sh \
 R_real_M16 "$COMMON --n-prefix 16" \
 R_1M_M16 "$COMMON --n-prefix 16 --catalog $C/prefixes_1000000.txt" \
 R_1M_M32 "$COMMON --n-prefix 32 --catalog $C/prefixes_1000000.txt" \
 R_300kS_M32 "$COMMON --n-prefix 32 --catalog $C/siblings_300000.txt" \
 R_real_M16b "$COMMON --n-prefix 16"
