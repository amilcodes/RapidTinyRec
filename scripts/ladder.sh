#!/bin/bash
# Final ladder: every rung in one allocation, same GPU, latency only, two passes each.
bash scripts/run_cfgs.sh \
 L0_fp32_graph "--dtype float32 --modes latency --repeats 2" \
 L1_bf16_graph "--dtype bfloat16 --modes latency --repeats 2" \
 L2_bf16_sep "--dtype bfloat16 --suffix separate --modes latency --repeats 2" \
 L3_bf16_compile "--dtype bfloat16 --compile default --modes latency --repeats 2" \
 L4_bf16_ff_tc "--dtype bfloat16 --fuse-first --attn triton --compile default --modes latency --repeats 2" \
 L5_ex3_nosuf "--engine exact3 --suffix none --n-prefix 64 --n-expand 50 --fuse-first --dtype bfloat16 --attn triton --compile default --modes latency --repeats 2" \
 L6_ex3_trie "--engine exact3 --norm trie --n-prefix 64 --n-expand 50 --fuse-first --dtype bfloat16 --attn triton --compile default --modes latency --repeats 2" \
 L7_ex3_trie_m32 "--engine exact3 --norm trie --n-prefix 32 --n-expand 50 --fuse-first --dtype bfloat16 --attn triton --compile default --modes latency,quality --repeats 2" \
 L8_exff_full_n16 "--engine exact --n-expand 16 --fuse-first --dtype bfloat16 --attn triton --compile default --modes latency --repeats 2" \
 L9_beam_trie_ff_tc "--norm trie --dtype bfloat16 --fuse-first --attn triton --compile default --modes latency,quality --repeats 2" \
 L10_beam_nosuf_ff_tc "--suffix none --dtype bfloat16 --fuse-first --attn triton --compile default --modes latency,quality --repeats 2" \
 $EXTRA
