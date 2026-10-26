# Complete TF32 paired pilot, October26

Job **7973479** completed the frozen fresh-process FP32/TF32/TF32/FP32 schedule in **18m17s**. The saved audit passed control historical exactness, within-policy stability/ownership, actual graph-linked TF32 dispatch, validity, release/source/config identity and four-phase log recovery. All predeclared latency gates passed.

Generation median/p95 fell **115.575893/118.961386 → 82.956653/83.654502 ms** (**28.22%/29.68% lower**) across24 paired requests from4 unique users. This is generation latency, not serving throughput. No cross-run compounded speedup or logging-causality claim.

| Stratum | FP32 median / p95 ms | TF32 median / p95 ms |
| :--- | ---: | ---: |
| Overall | 115.575893 / 118.961386 | 82.956653 / 83.654502 |
| Length 76 | 111.644270 / 113.096250 | 81.625141 / 82.473753 |
| Length 121 | 118.335681 / 119.001502 | 83.487649 / 83.693563 |
| Half 0 | 115.103392 / 118.599888 | 82.956653 / 83.501797 |
| Half 1 | 115.738383 / 119.001502 | 82.892248 / 83.693563 |
| Length:half 76:0 | 111.800438 / 112.556107 | 81.603428 / 82.349243 |
| Length:half 76:1 | 111.548347 / 113.349356 | 81.690373 / 82.262181 |
| Length:half 121:0 | 118.185787 / 118.687495 | 83.451774 / 83.511154 |
| Length:half 121:1 | 118.477333 / 119.004612 | 83.558930 / 83.710323 |

TF32 scores differ: **1182/1200 repeated rank positions** match (**197/200 unique**); no sequence/transition values match bitwise. Maximum common sequence/transition differences are0.009477615356445312/0.004464149475097656. User6 changes three rank positions; its held-out target remains rank45. Other target ranks remain absent/1/absent for users1/2/10. All four users have unchanged Hit/NDCG/MRR@1/5/10/20/50, with no invalid or duplicate outputs. NDCG@10 and Hit@10 are0.25 for both policies. Catalog coverage is0.0557491 for both.

**Four users cannot establish population quality parity.** The zero paired bootstrap intervals arise from resampling four observed zero metric changes. They do not establish noninferiority. These are SID-level results; all four pilot targets are unambiguous, while7 of500 full-cohort targets are ambiguous. Candidate numerical drift remains preserved.

| Phase | Policy/cache | Setup s | Nested compiler s | Nested capture/warmup s | Allocated / reserved bytes | Measurement peak allocated bytes |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | folded_fp32 / cold | 349.798 | 263.751 | 214.260 | 6595040256 / 8313110528 | 6913109504 |
| 1 | folded_tf32 / cold | 354.590 | 265.529 | 217.167 | 6595040256 / 8313110528 | 6913109504 |
| 2 | folded_tf32 / reuse | 152.596 | 67.859 | 86.185 | 6586438656 / 8539602944 | 6908346880 |
| 3 | folded_fp32 / reuse | 147.237 | 67.098 | 82.129 | 6595040256 / 8552185856 | 6916948480 |

Compiler and capture intervals nest in setup and must not be added. Model loading is about1.577s per phase. The child exited normally after1094.222s under1620s; all four tmpfs log publications/recoveries completed with no leftovers. Setup peak allocated bytes are8411338752/8411338752/7858849280/7867450880. Each phase has6176878592 unique/logical model bytes,15546688 unique input storage bytes versus79599936 logical bytes,6938624 prefix capacity bytes and8601600 suffix capacity bytes. Full memory reports and per-user paired timing changes are in the audit/review.

Eligible to **design a separate full500 quality review**; no full500 job/profile is ready or automatically submitted. Promotion and noninferiority remain false. The accepted baseline and PR17 retain their exact FP32 contracts.

The complete archive contains538 verified members, including447 raw files (289747256 bytes),75 submitted files, config, source/build/resource/cohort records, logs, generated code, traces, frozen audit, CI log, one-time guard and same-agent review. The review is a same-agent cross-check, not an independent reviewer. Source0ad2adfcec88fe8dd31b7c7fdaeda425e8c9c98b passed113CPUtests/CLI before submission. Reproduction instructions are archived; verify every shard/member against archive-manifest.json and preserve existing outputs. Earlier timeout and setup diagnostic evidence remain immutable.
