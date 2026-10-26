#!/bin/bash
set -euo pipefail
test "$#" -eq 0
test -n "${SLURM_JOB_ID:-}"
test -n "${OSCAR_RUN_DIR:-}"
test -d "${OSCAR_OUTPUT_DIR:-}"
# Original source-manifest.json and job-id remain at the OSCAR run root.
# The supervised coordinator checks their actual bytes before model loading.
cd "$OSCAR_RUN_DIR/code"
export TORCHINDUCTOR_COMPILE_THREADS=4
python_bin=/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python
exec "$python_bin" -B -m rapidtinyrec.serving_runner \
    --run-root "$OSCAR_RUN_DIR" \
    --release-root /oscar/scratch/aagraw73/minionerec-shared-prefill/release \
    --historical /oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users
