#!/bin/bash
# Experiments asked for by the first external review (paper §4, Appendix). One H100 allocation per section.
# usage (inside oscar.sh submit): bash scripts/review_job.sh ctlA|ctlB|ver1|ver2|ver3|sclW|sclL|sclF
# (H100s are on gpu-debug, < 1 h per job: each objective's comparison stays inside one job.)
#   ctl  controlled comparison: same engine, FP16, same objective. cbtrie objective: beam-50, certified at full depth,
#        + standalone third-code bridge (22), + both bridges (18/26, deployed). FlashRec's objective (codebook, no suffix): beam-50, certified (exact3),
#        + third-code bridge. MiniOneRec semantics beam (latency) for reference. Latency, all-user quality, throughput.
#   ver  verification: certified (full depth / bridged) vs the same objective scored for EVERY item. Bridged
#        exhaustive = the same engine with all 581 multi-child prefixes expanded (pool = whole catalog).
#        Exhaustive FP16 and FP32 checks of the full-depth decoder (500 cohort users, scorer with matching weights).
#   scl  serving boundary: wider first level (88..600 first codes, synthetic, latency only), longer prompts
#        (+1k / +3k tokens of history), and fusion on a 1M-SID catalog (random per-item term, scatter-max included).
set -u
S=/oscar/scratch/aagraw73/speedrun
BR18=18:$S/runs/bridgeBW/out/bridge/bridge_b_e18_w.pt
BR26=26:$S/runs/bridgeC26/out/bridge/bridge_e26_cas.pt     # cascade: trained on top of BR18, use together
BR22=22:$S/runs/bridgeW/out/bridge/bridge_e22_w.pt          # standalone third-code bridge (no first-level bridge)
BASE="--fuse-first --template-cache --attn triton --compile default --pdl --combo --dtype float16 --repeats 2"
CB="--norm cbtrie --weights 1,0.75,0"
CERT="--engine exacttrie --n-prefix 16 --n-expand 50 --tree-fast --fuse-store --compile-layer"
TP="--throughput-batches 16,32,64 --throughput-users 1024"
case $1 in
ctlA)
  bash scripts/run_cfgs.sh \
    X1_beam_cbtrie   "--engine beam $CB $BASE --modes latency,quality,throughput $TP" \
    X2_cert_full     "$CERT $CB $BASE --modes latency,quality,throughput $TP" \
    X3_cert_c22      "$CERT $CB $BASE --late-entry $BR22 --modes latency,quality" \
    X4_cert_18_26    "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency,quality,throughput $TP"
  ;;
ctlB)
  bash scripts/run_cfgs.sh \
    Y1_beam_cb       "--engine beam --norm codebook --suffix none $BASE --modes latency,quality,throughput $TP" \
    Y2_cert_cb       "--engine exact3 --norm codebook --n-prefix 64 --n-expand 50 --tree-fast --compile-layer $BASE --modes latency,quality,throughput $TP" \
    Y3_cert_cb_c22   "--engine exact3 --norm codebook --n-prefix 64 --n-expand 50 --tree-fast --compile-layer $BASE --late-entry $BR22 --modes latency,quality" \
    Z1_beam_full     "--engine beam $BASE --modes latency"
  ;;
ver1)
  bash scripts/run_cfgs.sh \
    V1_full_M16      "$CERT $CB $BASE --modes latency,quality --check-exhaustive 500" \
    V2_full_all      "--engine exacttrie --n-prefix 581 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --modes quality"
  ;;
ver2)
  bash scripts/run_cfgs.sh \
    V3_br_M16        "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency,quality" \
    V4_br_all        "--engine exacttrie --n-prefix 581 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency,quality"
  ;;
ver3)
  bash scripts/run_cfgs.sh \
    V5_full_fp32     "--engine exacttrie --n-prefix 16 --n-expand 50 --tree-fast $CB --fuse-first --template-cache --attn triton --compile default --dtype float32 --repeats 1 --modes latency --check-exhaustive 500"
  ;;
sclW)
  CATF=$REL/$($PY -c 'from rapidtinyrec.speedrun import data; print(data.CATALOG)')
  $PY scripts/wide_catalog.py --catalog $CATF --out $OUT/wide --widths 88,176,352,600
  args=()
  for w in 88 176 352 600; do
    C=$OUT/wide/wide_A$w.txt
    args+=(W_cert_A$w "$CERT --norm trie --catalog $C $BASE --modes latency")
    args+=(W_beam_A$w "--engine beam --norm trie --catalog $C $BASE --modes latency")
  done
  bash scripts/run_cfgs.sh "${args[@]}" \
    W_beam_nofuse    "--engine beam --norm trie --attn triton --compile default --dtype float16 --repeats 2 --modes latency"
  ;;
sclL)
  bash scripts/run_cfgs.sh \
    L_cert_0         "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency" \
    L_beam_0         "--engine beam $CB $BASE --modes latency" \
    L_cert_1k        "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency --extra-context 1024" \
    L_beam_1k        "--engine beam $CB $BASE --modes latency --extra-context 1024" \
    L_cert_3k        "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency --extra-context 3072" \
    L_beam_3k        "--engine beam $CB $BASE --modes latency --extra-context 3072"
  ;;
sclF)
  C1M=$S/runs/scaleA/out/catalogs/prefixes_1000000.txt
  bash scripts/run_cfgs.sh \
    F_real_M16       "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency" \
    F_real_M50       "--engine exacttrie --n-prefix 50 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency" \
    F_real_full_M50  "--engine exacttrie --n-prefix 50 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --modes latency" \
    F_real_full_all  "--engine exacttrie --n-prefix 581 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --modes latency" \
    F_real_br_all    "--engine exacttrie --n-prefix 581 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency" \
    F_real_M16_v     "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency --random-bonus 0.5" \
    F_1M_M16         "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency --catalog $C1M" \
    F_1M_M16_v       "$CERT $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency --catalog $C1M --random-bonus 0.5" \
    F_1M_M32         "--engine exacttrie --n-prefix 32 --n-expand 50 --tree-fast --fuse-store --compile-layer $CB $BASE --late-entry-a $BR18 --late-entry $BR26 --modes latency --catalog $C1M"
  ;;
esac
