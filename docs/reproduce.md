# Reproduce

Use Python 3.11 on Linux and an Ampere CUDA GPU. The published measurements used
an RTX A6000, PyTorch 2.14.0+cu126, Transformers 4.55.4, four CPU threads and FP32
with TF32 disabled. GPU inference needs a compute allocation on shared clusters.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
python -m rapidtinyrec.prepare_release --release-root releases/office
python -m rapidtinyrec.run prefill --release-root releases/office \
  --dtype float32 --output-root results/prefill
python -m rapidtinyrec.audit results/prefill/shared_prefill_500_users \
  --output results/prefill/audit.json
```

The downloader pins and checks every model, tokenizer and dataset file. The
benchmark selects the first 500 distinct test users in file order and verifies
the complete prompt/cohort hash before loading the model. The upstream dataset
reader is used only with this hash-verified release.

Output directories contain manifests, per-request token sequences and scores,
and timing summaries. These generated files stay outside Git. Use a fresh output
directory per experiment. Never run two benchmark processes on the same GPU.

`python -m unittest discover -s tests -p test_evidence.py -v` verifies the
published per-request measurements using only the Python standard library.

## Other ablations

```bash
python -m rapidtinyrec.run batched --release-root releases/office --output-root results/batched
python -m rapidtinyrec.run gpu --release-root releases/office --output-root results/gpu
python -m rapidtinyrec.run compiled --release-root releases/office --output-root results/compiled
python -m rapidtinyrec.run gqa --release-root releases/office \
  --reference-results results/compiled/compile_forward_500_users --output-root results/gqa
python -m rapidtinyrec.run decode-pilot --release-root releases/office \
  --reference-results results/gqa/gqa_500_users --output-root results/replay-pilot \
  --direct-reorder --repeats 6
python -m rapidtinyrec.run decode-pilot --release-root releases/office \
  --reference-results results/gqa/gqa_500_users --output-root results/replay-full \
  --direct-reorder --full-cohort --repeats 1
```

`compiled` and `gqa` each create a fresh compiler-cache directory before importing
PyTorch. Compilation and warmup are saved separately. GQA compares eager,
original compiled and GQA-specialized compiled forwards in the same process;
its second compiled variant has an incremental, not independently cold, startup.
The GQA command requires an A6000 to preserve the published hardware comparison.

Every mode produces `summary.json` and supports the standalone `audit` command
with its respective `*_500_users` result directory. The GQA output also includes
`compile_timing.json`. To inspect CLI options directly, run e.g.
`python -m rapidtinyrec.benchmark_gqa --help`.

Replay has a separate provenance/bit audit. Preserve the submitted source and
run record, then run:

```bash
python -m rapidtinyrec.audit_decode_pilot results/replay-full/decode_replay_500 \
  --historical results/gqa/gqa_500_users \
  --snapshot frozen-source/rapidtinyrec --run-record run-record.json \
  --output results/replay-full/audit.json
```

Run the bounded pilot before the full cohort. The full run warms all 40 graph
shapes and checks memory before measurement. [Protocol and record format](decode-replay.md).

## Profile

```bash
python -m rapidtinyrec.run profile-shared --release-root releases/office \
  --reference-results results/prefill/shared_prefill_500_users --output-root results/profile-shared
python -m rapidtinyrec.run profile-compiled --release-root releases/office \
  --reference-results results/compiled/compile_forward_500_users \
  --output-root results/profile-compiled --mask-ablation
```

Profiling covers representatives of the ten history lengths. Uninstrumented
latency runs precede scope timing and CUDA tracing. Stream-event spans include
idle time; use trace kernel activities to distinguish GPU work from launch gaps.
Profiler timings are not the 500-user benchmark numbers.

The [input-arena protocol](input-arena.md) compares direct reorder against shared
input storage in isolated processes, with separate memory and setup measurements.

The [physical-prefix protocol](shared-prefix.md) compares shared inputs against
one stored history KV row per layer. Run its pilot first, then all four full
phases in one 30-minute A6000 allocation; the completed comparison took 16m07s.
The full audit requires all 500 paired controls and never accepts partial output.
