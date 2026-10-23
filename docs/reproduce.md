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
python -m rapidtinyrec.benchmark --release-root releases/office \
  --dtype float32 --output-root results/prefill
python -m rapidtinyrec.audit_results results/prefill/shared_prefill_500_users \
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
