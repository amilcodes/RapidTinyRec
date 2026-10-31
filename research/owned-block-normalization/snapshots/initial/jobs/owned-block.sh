#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "${OSCAR_RUN_DIR:?}/code"
exec /oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python -u allocated_entry.py outer
