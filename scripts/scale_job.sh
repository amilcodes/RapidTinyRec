#!/bin/bash
# Catalog scaling: certification / expansion sets on synthetic catalogs, then latency at fixed budgets.
S=/oscar/scratch/aagraw73/speedrun/runs
BR="--late-entry-a 18:$S/bridgeBW/out/bridge/bridge_b_e18_w.pt --late-entry 26:$S/bridgeC26/out/bridge/bridge_e26_cas.pt"
$PY scripts/scale_catalog.py --release $REL --out $OUT $BR 2>&1 | grep --line-buffered -E '^\{"stage"|Error|Traceback|error' | cut -c1-4000
COMMON="--engine exacttrie --norm cbtrie --weights 1,0.75,0 --attn triton --fuse-first --template-cache --tree-fast \
 --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 $BR --modes latency --repeats 1"
for c in siblings_100000 siblings_300000 prefixes_100000 prefixes_1000000; do
  for M in 16 64; do
    EXTRA= bash scripts/run_cfgs.sh L_${c}_M$M "$COMMON --catalog $OUT/catalogs/$c.txt --n-prefix $M --n-expand 50"
  done
done
