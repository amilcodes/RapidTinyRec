#!/bin/bash
# usage: sft_job.sh TAG "train_sft args"   (Office; evaluates the fine-tuned model with the certified decoder)
set -u
TAG=$1; ARGS=$2
SR=/oscar/scratch/aagraw73/speedrun
EV="--engine exacttrie --n-prefix 16 --n-expand 50 --norm cbtrie --weights 1,0.75,0 --fuse-first --template-cache --dtype float16 --attn triton --tree-fast --modes quality --quality-batch 32"
$PY scripts/train_sft.py --release $REL --train-release $SR/release_office_train --valid-release $SR/release_office_valid --out $OUT/$TAG $ARGS 2>&1 | grep --line-buffered -E '"stage"|Error|error|Traceback' | grep --line-buffered -v '"step": [0-9]*[1-9]0,\|"step": [0-9]*[1-9]50,' | cut -c1-400
for split in valid test; do
  $PY scripts/bench.py --release $OUT/$TAG/ckpt_$split --out $OUT/$TAG/ev_$split $EV > $OUT/$TAG/ev_$split.log 2>&1
  echo "=== EVAL $TAG $split rc=$?"
  grep -E '"stage": "quality"|Traceback|Error' $OUT/$TAG/ev_$split.log | cut -c1-600
done
