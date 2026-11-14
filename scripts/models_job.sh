#!/bin/bash
# Other backbones with the released MiniOneRec SID index (same 600 SID tokens, same catalog and test split):
# community SFT checkpoints of Qwen2.5-3B / 7B trained with MiniOneRec's script (HF acdxuan/minionerec).
# usage: models_job.sh PHASE MODEL_DIR EB EC [EPOCHS]
#   PHASE A: release-like dirs, beam-50 + certified full-depth baselines, bridge training (c, b, cascade c)
#   PHASE B: certified + bridges, + SASRec term (in the graph), FP8 throughput
#   EB / EC: first-level / third-code entry layers (scaled from 18 / 22 of 28 layers)
P=$1; MD=$2; EB=$3; EC=$4; EP=${5:-6}
S=/oscar/scratch/aagraw73/speedrun
TAG=$(basename $(dirname $OUT))
R=$S/runs/models_rel/$TAG
mkdir -p $R/test $R/valid
ln -sfn $REL/Amazon $R/test/Amazon; ln -sfn $MD $R/test/Office_ckpt
ln -sfn $S/release_office_valid/Amazon $R/valid/Amazon; ln -sfn $MD $R/valid/Office_ckpt
export REL=$R/test
BD=$S/runs/${TAG%_B}_A/out/bridge
COMMON="--engine exacttrie --norm cbtrie --weights 1,0.75,0 --attn triton --fuse-first --template-cache --tree-fast \
 --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --n-prefix 16 --n-expand 50"
G="grep --line-buffered -E '\"stage\": \"(bridge_eval|bridge_b_eval)\"|Error|Traceback'"
if [ "$P" = A ]; then
  bash scripts/run_cfgs.sh \
    M0_beam "--fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency,quality --repeats 1" \
    M1_trie "$COMMON --modes latency,quality --repeats 1"
  T="--train-release $R/valid --release $R/test --out $OUT/bridge --row-weight 0.5 --epochs $EP --train-users 4600"
  $PY scripts/train_bridge.py $T --entry $EC --tag _w 2>&1 | eval $G | cut -c1-900
  $PY scripts/train_bridge_b.py $T --entry $EB --tag _w --bridge-c $OUT/bridge/bridge_e${EC}_w.pt 2>&1 | eval $G | cut -c1-900
  $PY scripts/train_bridge.py $T --entry $EC --b-entry $EB --b-bridge $OUT/bridge/bridge_b_e${EB}_w.pt --tag _cas 2>&1 | eval $G | cut -c1-900
else
  BC="--late-entry $EC:$BD/bridge_e${EC}_w.pt"
  BR="--late-entry-a $EB:$BD/bridge_b_e${EB}_w.pt --late-entry $EC:$BD/bridge_e${EC}_cas.pt"
  bash scripts/run_cfgs.sh \
    M2c_conly "$COMMON $BC --modes latency,quality --repeats 1" \
    M2_br "$COMMON $BR --modes latency,quality --repeats 1" \
    M3_fus "$COMMON $BC --collab $S/runs/fuseC/out/sasrec --bonus-beta 0.5 --modes latency,quality --repeats 1" \
    M4_tp8 "$COMMON $BC --collab $S/runs/fuseC/out/sasrec --bonus-beta 0.5 --quant fp8 --modes throughput,quality --throughput-batches 32,64" \
    M5_beam "--fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency --repeats 1"
fi
