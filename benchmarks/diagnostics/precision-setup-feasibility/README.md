# TF32 setup feasibility result

Job **7972055** completed in **9m22s**. The frozen setup auditor passed: all **1,200 ranks, 1,200 FP32 sequence values and 6,000 transitions**, plus ancestry, match the preserved original TF32 outputs exactly. Both phases pass warm/direct/graph/repeat/retained-output stability, score-storage ownership, actual graph-linked TF32 dispatch at both lengths, and complete log recovery. All 69 submitted files and 227 raw files (144,986,577 bytes) are hash-verified; the archive contains 310 verified members, including source/config/resources, API logs, traces, generated code, CI log, audit and reproduction records.

| TF32 phase | Setup seconds | Nested backend compile seconds | Nested capture/warmup seconds | Generation median/p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Cold, phase 1 | 357.409 | 269.866 | 219.156 | 83.263 / 83.713 |
| Disk reuse, phase 2 | 152.681 | 67.777 | 86.640 | 82.868 / 83.734 |

Setup/compile/capture durations overlap and must not be added. Each fresh process captures eight model graphs, with no measured compilation/capture. The supervisor child completed normally in 556.635 seconds against its 1620-second deadline. Both phase logs were staged on tmpfs, disabled/flushed before measurement and copied with verified hashes; copy times were 0.316/0.292 seconds. Recovery found no leftover local entries. This normal exit does not demonstrate recovery after supervisor kill, node loss or lost child buffers.

Generation timings are descriptive only: overall 83.256/83.732 ms, length76 81.724/82.712 ms and length121 83.644/83.777 ms. All phase/user/length timings and allocated/reserved/peak/logical/unique storage capacities remain in summary.json and audit.json. Model parameter storage is 6,176,878,592 bytes per process; unique replay input storage 15,546,688 bytes versus logical 79,599,936 bytes; physical prefix/suffix capacities 6,938,624/8,601,600 bytes, output suffix storage 57,344,000 bytes.

This is **TF32-only setup feasibility**, with four users and 24 repeated requests. It contains no paired FP32 control, new held-out quality estimate, causal logging comparison or completed ABBA performance gate. Historical FP32 scores differ and 1182/1200 repeated ranks match; those fidelity differences are preserved. Full500 eligibility, promotion and noninferiority remain false. The original timeout and its failure evidence are untouched.

The separate post-audit review is by the same research agent, not an independent reviewer identity. It documents two limitations: generic manifest protocol prose still describes ABBA although the experiment discriminator/frozen schedule/actual phase directories specify TF32-only phases1/2; the setup schema emits no new independent on-node checkpoint byte digest, so release identity relies on pinned references/loading source and exact prior-policy outputs. Preserve both caveats.

The result supports designing a separately frozen complete paired pilot. Doubling this TF32 cold/reuse child duration gives about1113 seconds as a planning estimate, not a completion guarantee: repaired-logging FP32 setup remains unmeasured. Review conservative phase budgets and preserve the1620-second deadline/cleanup and failure evidence before any one-time submission. No automatic retry or full500 run follows this diagnostic. The original paired ≥10% median/p95 performance gate and quality requirements remain unchanged.
