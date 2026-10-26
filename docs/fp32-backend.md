# Native FP32 BLAS backend diagnostic

This is a bounded fixed-input diagnostic, not an inference optimization result. It compares unchanged default dispatch, explicit cuBLAS as a control, and cuBLASLt preference for original-shaped gate/up/down `mm.out` projections. No packing, altered dimensions/layouts, fusion, workspace tuning, algorithm sweep or precision change is included. The source starts from accepted folded/profile PR12 and contains none of the rejected MLP/constraint/beam candidates.

The accepted diagnostic profile attributes 27.788–27.933 ms of GPU activity to decode gate/up and 14.554–14.850 ms to down projections at lengths76/101/121. Instrumented activity is not predicted recoverable latency. The earlier wide/batched projection diagnosis changed FP32 bits, motivating a test that keeps projection shapes intact.

PyTorch2.14's [preference API](https://docs.pytorch.org/docs/2.14/backends.html#torch.backends.cuda.preferred_blas_library) can request cuBLASLt but can fall back. Pinned commit `2b3ec34829036a65cd9d1398ea72a0167dc37470` [dispatches float GEMMs](https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/aten/src/ATen/cuda/CUDABlas.cpp#L1226) through Lt when selected. Its getter resolves `Default` to `Cublas` on NVIDIA; the diagnostic records requested and resolved preferences separately. Runtime build, FP32/TF32off and disabled tuning are checked. API logs and profiler traces establish dispatch separately from bit equality; a preference getter or kernel name alone cannot certify it.

The fixed cohort is users1/2/6/10 at prompt lengths76/121, layers0/13/27. First-decode gate/up inputs and **independent original down inputs** are captured from eager folded-attention/shared-prefill generation under default dispatch. They are representative eager inputs, not reconstructed compiled intermediates. Gate/up inputs are `[50,1536]`, weights `[8960,1536]`; down inputs `[50,8960]`, weights `[1536,8960]`. Each weight is transposed as a view and outputs retain original contiguous layouts. Checkpoint/cohort hashes and weight digests are recorded.

Nine independent graphs per case (three roles × three settings) use owned staging and output storage on one sequential nondefault CUDA stream after activation collection completes. GPU checks save direct/direct-repeat, graph/graph-repeat, retained-after-other-role, changed-input and restored-input outputs, plus post-timing outputs. Every raw FP32 tensor is content-addressed and gzip-compressed; identical data is stored once while every observation retains its reference. Signed zeros and nonfinite values are preserved. No tolerance is accepted. An unexpected default/explicit-cuBLAS difference blocks interpretation.

Each role/setting has eight CUDA-event samples averaging32 graph replays, with setting order reversed each round. Every sample is retained. Timing excludes backend selection, capture/warmup, staging, logging/profiling and evidence extraction, and measures only projection graph replay. It is **not generation-call latency**. Setup, model loading, per-case capture, allocation/reservation peaks and logical/unique storage are separate; nested setup times must not be added. No model compilation or capture occurs in measured loops. cuBLAS/Lt API logging is enabled only for untimed dispatch probes.

The independent standard-library auditor rechecks raw binary hashes/bits, shape/stride/scalar metadata, repeated/changed/restored inputs, independent storage, all36 original projection cases, every timing sample, source and checkpoint provenance. Mismatches are preserved before nonzero exit. Missing API logs or observed fallback prevents verified backend-change claims even when scores agree. Synthetic tests cover signed-zero differences, nonfinite values, backend restoration, corruption, stale graphs, aliases, missing cases, control drift and failure preservation. CPU checks do not prove CUDA correctness.

Run only on one allocated regular A6000,4CPUs32GB with a10-minute bound:

```bash
bash jobs/fp32-backend.sh
python3 -m rapidtinyrec.audit_fp32_backend RAW/outputs/diagnosis \
  --snapshot FROZEN_SOURCE --run-record RUN_RECORD.json --output audit.json
```

The source/config/resources are frozen before submission; an exclusive intent and saved run/job identity prevent duplicate submissions. After terminal completion fetch immutable outputs and run the exact saved audit argv/cwd. Raw evidence may be published in hash-verified shards below50MiB; failures will not be omitted.

Only audited exact/stable outputs, verified meaningful dispatch change and useful consistent microtiming benefit justify a separately scoped paired generation pilot. A global BLAS preference also affects prefill/attention/head operations and needs explicit end-to-end validation. This diagnostic cannot authorize full500 or a speedup claim. A future inference candidate must still pass paired/historical rank, sequence and all five transition bits plus useful generation-call median/p95 by length/half. If this bounded diagnostic fails, publish it and reassess the score-fidelity/quality tradeoff instead of expanding an unbounded search.
