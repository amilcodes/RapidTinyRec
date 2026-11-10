#!/bin/bash
# Catalog scaling on Industrial (48 first codes): certification / expansion sets, then same-allocation latency.
export SPEEDRUN_DATASET=industrial REL=/oscar/scratch/aagraw73/speedrun/release_ind
S=/oscar/scratch/aagraw73/speedrun/runs
BR="--late-entry-a 18:$S/casInd/out/bridge_ind/bridge_b_e18_w.pt --late-entry 18:$S/casInd/out/bridge_ind/bridge_e18_cas.pt"
$PY scripts/scale_catalog.py --release $REL --out $OUT $BR --sizes 10000,100000,300000,1000000 --Ms 16,32,64,128 2>&1 | grep --line-buffered -E '^\{"stage"|Error|Traceback|error' | cut -c1-1500
COMMON="--engine exacttrie --norm cbtrie --weights 1,0.75,0 --attn triton --fuse-first --template-cache --tree-fast \
 --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 $BR --modes latency --repeats 2 --n-expand 50"
bash scripts/run_cfgs.sh \
 R_real_M16 "$COMMON --n-prefix 16" \
 R_1M_M16 "$COMMON --n-prefix 16 --catalog $OUT/catalogs/prefixes_1000000.txt" \
 R_1M_M32 "$COMMON --n-prefix 32 --catalog $OUT/catalogs/prefixes_1000000.txt" \
 R_real_M16b "$COMMON --n-prefix 16"
