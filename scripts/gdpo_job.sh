#!/bin/bash
# SFT vs RL (GDPO) on the same backbone and SID index (HF onesfour/MiniOneRec-1.5B-SFT-GDPO, Industrial,
# own RQ-VAE index and splits), both in one allocation. The index gives 19% of items a 4th code to separate
# collisions; SPEEDRUN_SID_LEVELS=3 merges each collision group into its 3-code SID (268 groups). Per model:
# beam-50 and certified exact under the SAME objective (trie normalization, no forced suffix: beam's search
# error), and the fast certified decoder (cbtrie 0.75, M = 16).
export SPEEDRUN_DATASET=industrial SPEEDRUN_SID_LEVELS=3
H=/oscar/scratch/aagraw73/speedrun/hf/onesfour
for m in sft gdpo; do
  case $m in sft) CK=$H/sft_model ;; gdpo) CK=$H/gdpo_best_checkpoint-1155 ;; esac
  R=/oscar/scratch/aagraw73/speedrun/runs/models_rel/gdpo_$m
  mkdir -p $R/test $R/valid/Amazon/test
  ln -sfn $H/data $R/test/Amazon; ln -sfn $CK $R/test/Industrial_ckpt
  ln -sfn $H/data/info $R/valid/Amazon/info; ln -sfn $CK $R/valid/Industrial_ckpt
  ln -sfn $H/data/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv $R/valid/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
  export REL=$R/test
  bash scripts/run_cfgs.sh \
   ${m}_beam_trie "--norm trie --suffix none --fuse-first --template-cache --dtype float16 --attn triton --compile default --modes latency,quality --repeats 1" \
   ${m}_exact_trie "--engine exacttrie --norm trie --n-prefix 16 --n-expand 50 --fuse-first --template-cache --tree-fast --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --attn triton --modes latency,quality --repeats 1" \
   ${m}_cbtrie "--engine exacttrie --norm cbtrie --weights 1,0.75,0 --n-prefix 16 --n-expand 50 --fuse-first --template-cache --tree-fast --fuse-store --compile default --compile-layer --pdl --combo --dtype float16 --attn triton --modes latency,quality --repeats 1"
done
