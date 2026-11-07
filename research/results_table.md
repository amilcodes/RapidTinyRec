| run | args | median ms | p95 ms | top1=FP32 | top10 set=FP32 | HR@5 | HR@10 | NDCG@10 | HR@50 | cert@1/5/10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sweepA/bf16_compile | beam bfloat16 full suf=fused q=none c=default attn=torch | 35.21 | 35.56 | 465 | 158 | 0.1414 | 0.1578 | 0.1245 | 0.2361 |  |
| sweepA/bf16_nosuffix | beam bfloat16 full suf=none q=none c=none attn=torch | 30.32 | 30.84 | 329 | 0 | 0.1436 | 0.1624 | 0.1284 | 0.2363 |  |
| sweepA/bf16_profile | beam bfloat16 full suf=fused q=none c=none attn=torch |  |  |  |  |  |  |  |  |  |
| sweepA/bf16_sep | beam bfloat16 full suf=separate q=none c=none attn=torch | 51.26 | 51.97 | 471 | 175 |  |  |  |  |  |
| sweepA/fp32 | beam float32 full suf=fused q=none c=none attn=torch | 98.78 | 103.09 | 500 | 500 | 0.1404 | 0.1589 | 0.1247 | 0.2361 |  |
| sweepB/ex_bf16_n128 | exact bfloat16 full suf=fused q=none c=none attn=torch N=128 M= | 45.50 | 46.32 | 470 | 113 | 0.1416 | 0.1587 | 0.1244 | 0.2285 | 0.99/0.75/0.54 |
| sweepB/ex_bf16_n16 | exact bfloat16 full suf=fused q=none c=none attn=torch N=16 M= | 31.11 | 31.62 | 454 | 0 | 0.1469 | 0.1658 | 0.1284 | 0.1743 | 0.88/0.11/0.00 |
| sweepB/ex_bf16_n32 | exact bfloat16 full suf=fused q=none c=none attn=torch N=32 M= | 32.40 | 32.92 | 468 | 26 | 0.1432 | 0.1652 | 0.1270 | 0.2018 | 0.96/0.32/0.05 |
| sweepB/ex_bf16_n64 | exact bfloat16 full suf=fused q=none c=none attn=torch N=64 M= | 36.56 | 37.12 | 472 | 77 | 0.1426 | 0.1599 | 0.1251 | 0.2316 | 0.98/0.58/0.27 |
| sweepB/ex_bf16_nosuf_n64 | exact bfloat16 full suf=none q=none c=none attn=torch N=64 M= | 31.47 | 31.99 | 328 | 0 | 0.1432 | 0.1624 | 0.1284 | 0.2343 | /0.94/0.83 |
| sweepB/ex_fp32_n64 | exact float32 full suf=fused q=none c=none attn=torch N=64 M= | 100.81 | 104.91 | 497 | 186 | 0.1422 | 0.1595 | 0.1251 | 0.2310 | 0.99/0.60/0.29 |

Paired bootstrap (2,000 resamples over 4,866 test users) vs `sweepA/fp32`:

| run | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] | ΔHR@5 [95% CI] |
|---|---|---|---|
| sweepA/bf16_compile | -0.0010 [-0.0027, +0.0006] | -0.0003 [-0.0011, +0.0006] | +0.0010 [-0.0008, +0.0029] |
| sweepA/bf16_nosuffix | +0.0035 [-0.0008, +0.0076] | +0.0036 [+0.0013, +0.0061] | +0.0033 [-0.0008, +0.0074] |
| sweepB/ex_bf16_n128 | -0.0002 [-0.0023, +0.0021] | -0.0003 [-0.0012, +0.0006] | +0.0012 [-0.0004, +0.0031] |
| sweepB/ex_bf16_n16 | +0.0070 [+0.0012, +0.0129] | +0.0036 [+0.0013, +0.0060] | +0.0066 [+0.0029, +0.0105] |
| sweepB/ex_bf16_n32 | +0.0064 [+0.0021, +0.0109] | +0.0023 [+0.0007, +0.0040] | +0.0029 [+0.0002, +0.0058] |
| sweepB/ex_bf16_n64 | +0.0010 [-0.0014, +0.0035] | +0.0004 [-0.0007, +0.0015] | +0.0023 [+0.0004, +0.0043] |
| sweepB/ex_bf16_nosuf_n64 | +0.0035 [-0.0008, +0.0076] | +0.0036 [+0.0012, +0.0061] | +0.0029 [-0.0012, +0.0070] |
| sweepB/ex_fp32_n64 | +0.0006 [-0.0016, +0.0027] | +0.0003 [-0.0004, +0.0010] | +0.0018 [+0.0004, +0.0033] |

