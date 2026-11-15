#!/bin/bash
# Do late-entry bridges get closer to the full-depth model with more (label-free) training prompts?
# The deployed bridges were distilled on 4,600 validation prompts. Distillation uses no labels, so the training
# split's prompts are fair game too. Same recipe (row weight 0.5, 6 epochs), N training prompts from the train split.
# usage (inside oscar.sh submit): bash scripts/bridge_data_job.sh b N | c N B_BRIDGE.pt | ceval N B_BRIDGE.pt | eval B C TAG
#   b     first-level bridge at layer 18 (the main source of disagreement with full depth)
#   c     cascade third-code bridge at layer 26 on top of a given first-level bridge
#   eval  deployed configuration with the given bridges: latency + all-user quality (+ lists for agreement)
set -u
S=/oscar/scratch/aagraw73/speedrun
TR=$S/release_office_train
G='"stage": "(ridge|train|bridge_eval|bridge_b_eval)"|Error|Traceback|line [0-9]'
case $1 in
b)
  $PY scripts/train_bridge_b.py --train-release $TR --release $REL --out $OUT/bridge --entry 18 --row-weight 0.5 \
    --epochs 6 --train-users $2 --tag _wT$2 --bridge-c $S/runs/bridgeW/out/bridge/bridge_e22_w.pt 2>&1 \
    | grep -E "$G" | cut -c1-1500
  ;;
c)
  $PY scripts/train_bridge.py --train-release $TR --release $REL --out $OUT/bridge --entry 26 --b-entry 18 \
    --b-bridge $3 --row-weight 0.5 --epochs 6 --train-users $2 --tag _casT$2 2>&1 | grep -E "$G" | cut -c1-1500
  ;;
ceval)
  # cascade third-code bridge on top of the given first-level bridge, then the deployed decoder with both
  B=$3
  $PY scripts/train_bridge.py --train-release $TR --release $REL --out $OUT/bridge --entry 26 --b-entry 18 \
    --b-bridge $B --row-weight 0.5 --epochs 6 --train-users $2 --tag _casT$2 2>&1 | grep -E "$G" | cut -c1-1500
  bash scripts/run_cfgs.sh \
    E_T$2 "--engine exacttrie --n-prefix 16 --n-expand 50 --tree-fast --fuse-store --compile-layer --norm cbtrie \
 --weights 1,0.75,0 --fuse-first --template-cache --attn triton --compile default --pdl --combo --dtype float16 \
 --repeats 2 --late-entry-a 18:$B --late-entry 26:$OUT/bridge/bridge_e26_casT$2.pt --modes latency,quality"
  ;;
eval)
  bash scripts/run_cfgs.sh \
    E_$4 "--engine exacttrie --n-prefix 16 --n-expand 50 --tree-fast --fuse-store --compile-layer --norm cbtrie \
 --weights 1,0.75,0 --fuse-first --template-cache --attn triton --compile default --pdl --combo --dtype float16 \
 --repeats 2 --late-entry-a 18:$2 --late-entry 26:$3 --modes latency,quality"
  ;;
esac
