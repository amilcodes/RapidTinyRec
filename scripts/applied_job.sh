#!/bin/bash
# Applied-setting tests (second external review): normalization, first-level breadth, long contexts.
# usage (inside oscar.sh submit): bash scripts/applied_job.sh hier|wide|fullA|fullB|ctx
#   hier   budget on the first level (HierTrieEngine, --engine hier): prompt pass -> exact s(a) for every first
#          code -> only the A1 best first codes become rows -> prefixes. A1 = 88 (all), 48, 32, 16 on Office
#          (latency, all-user quality, certification), deployed bridges.
#   wide   the same on synthetic catalogs with 600 first codes (latency only): two-forward vs hierarchical.
#   fullA  MiniOneRec's full-vocabulary objective served exactly (no forced suffix): beam-50 vs certified at
#          M = 64 / 256 / all 2,488 prefixes, full depth.
#   fullB  the same with the third-code bridge, and the forced suffix (MiniOneRec's exact objective).
#   ctx    long contexts: per-user history K/V cached (emulated by a long prefix shared by every user, held in the
#          template cache) at 1k / 4k / 8k tokens; throughput at 3k uncached tokens.
set -u
S=/oscar/scratch/aagraw73/speedrun
BR18=18:$S/runs/bridgeBW/out/bridge/bridge_b_e18_w.pt
BR26=26:$S/runs/bridgeC26/out/bridge/bridge_e26_cas.pt
BR22=22:$S/runs/bridgeW/out/bridge/bridge_e22_w.pt
BASE="--fuse-first --template-cache --attn triton --compile default --pdl --combo --dtype float16 --repeats 2"
CB="--norm cbtrie --weights 1,0.75,0"
FAST="--tree-fast --fuse-store --compile-layer"
DEP="--late-entry-a $BR18 --late-entry $BR26"
WIDE=$S/runs/rv_sclW/out/wide
case $1 in
hier)
  bash scripts/run_cfgs.sh \
    D_two_forward  "--engine exacttrie --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes latency,quality" \
    H_A88          "--engine hier --first-budget 88 --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes latency,quality --check-exhaustive 200" \
    H_A48          "--engine hier --first-budget 48 --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes latency,quality" \
    H_A32          "--engine hier --first-budget 32 --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes latency,quality" \
    H_A16          "--engine hier --first-budget 16 --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes latency,quality"
  ;;
wide)
  args=()
  for w in 88 176 352 600; do
    args+=(WD_two_A$w "--engine exacttrie --n-prefix 16 --n-expand 50 $FAST --norm trie --catalog $WIDE/wide_A$w.txt $BASE $DEP --modes latency")
    args+=(WH_b32_A$w "--engine hier --first-budget 32 --n-prefix 16 --n-expand 50 $FAST --norm trie --catalog $WIDE/wide_A$w.txt $BASE $DEP --modes latency")
  done
  bash scripts/run_cfgs.sh "${args[@]}"
  ;;
fullA)
  bash scripts/run_cfgs.sh \
    N1_beam_full   "--engine beam --norm full --suffix none $BASE --modes latency,quality" \
    N2_cert_M64    "--engine exact3 --norm full --suffix none --n-prefix 64 --n-expand 50 --tree-fast --compile-layer $BASE --modes latency,quality" \
    N3_cert_M256   "--engine exact3 --norm full --suffix none --n-prefix 256 --n-expand 50 --tree-fast --compile-layer $BASE --modes latency,quality" \
    N4_cert_all    "--engine exact3 --norm full --suffix none --n-prefix 2488 --n-expand 50 --tree-fast --compile-layer $BASE --modes latency,quality"
  ;;
fullB)
  bash scripts/run_cfgs.sh \
    N5_cert_M256_c22 "--engine exact3 --norm full --suffix none --n-prefix 256 --n-expand 50 --tree-fast --compile-layer $BASE --late-entry $BR22 --modes latency,quality" \
    N6_cert_all_c22  "--engine exact3 --norm full --suffix none --n-prefix 2488 --n-expand 50 --tree-fast --compile-layer $BASE --late-entry $BR22 --modes latency,quality" \
    N7_suf_M64       "--engine exact3 --norm full --suffix fused --n-prefix 64 --n-expand 64 --tree-fast --compile-layer $BASE --modes latency,quality" \
    N8_suf_M512      "--engine exact3 --norm full --suffix fused --n-prefix 512 --n-expand 512 --tree-fast --compile-layer $BASE --modes latency,quality"
  ;;
ctx)
  args=()
  for T in 1024 4096 8192; do
    args+=(C_cert_$T "--engine exacttrie --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes latency --shared-context $T")
    args+=(C_beam_$T "--engine beam $CB $BASE --modes latency --shared-context $T")
  done
  bash scripts/run_cfgs.sh "${args[@]}" \
    T_cert_3k  "--engine exacttrie --n-prefix 16 --n-expand 50 $FAST $CB $BASE $DEP --modes throughput --throughput-batches 4,8,16 --throughput-users 256 --extra-context 3072" \
    T_beam_3k  "--engine beam $CB $BASE --modes throughput --throughput-batches 4,8,16 --throughput-users 256 --extra-context 3072"
  ;;
esac
