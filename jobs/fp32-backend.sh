#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Allocated compute required}"
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_TUNABLEOP_TUNING=0
/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python -u -m rapidtinyrec.diagnose_fp32_backend --release /oscar/scratch/aagraw73/minionerec-shared-prefill/release --historical /oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users --output "$OSCAR_OUTPUT_DIR/diagnosis"
