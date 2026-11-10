#!/bin/bash
# SFT vs RL (GDPO) on the same backbone and SID index (HF onesfour/MiniOneRec-1.5B-SFT-GDPO, Industrial,
# own RQ-VAE index and splits). Per model: MiniOneRec beam-50 semantics, certified exact under the SAME objective
# (full vocabulary + forced suffix; search error of beam), and the fast certified decoder (cbtrie, M = 16).
# usage: gdpo_job.sh sft|gdpo
export SPEEDRUN_DATASET=industrial
H=/oscar/scratch/aagraw73/speedrun/hf/onesfour
case "$1" in sft) CK=$H/sft_model ;; gdpo) CK=$H/gdpo_best_checkpoint-1155 ;; esac
R=/oscar/scratch/aagraw73/speedrun/runs/models_rel/gdpo_$1
mkdir -p $R/test $R/valid/Amazon/test
ln -sfn $H/data $R/test/Amazon; ln -sfn $CK $R/test/Industrial_ckpt
ln -sfn $H/data/info $R/valid/Amazon/info; ln -sfn $CK $R/valid/Industrial_ckpt
ln -sfn $H/data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv $R/valid/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
export REL=$R/test
bash scripts/run_cfgs.sh \
 G0_beam "--fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency,quality --repeats 1" \
 G1_exact_full "--engine exact3 --norm full --suffix fused --n-prefix 64 --n-expand 50 --fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency,quality --repeats 1" \
 G2_trie "--engine exacttrie --norm cbtrie --weights 1,0.75,0 --n-prefix 16 --n-expand 50 --fuse-first --template-cache --tree-fast --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --attn triton --modes latency,quality --repeats 1" \
 G3_trie_plain "--engine exacttrie --norm trie --n-prefix 16 --n-expand 50 --fuse-first --template-cache --tree-fast --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --attn triton --modes quality"
