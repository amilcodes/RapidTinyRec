#!/usr/bin/env bash
set -euo pipefail
cd "$OSCAR_RUN_DIR/code"
PY=/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
scontrol show job -o "$SLURM_JOB_ID" > "$OSCAR_OUTPUT_DIR/scheduler-before.txt"
set +e
timeout --signal=TERM --kill-after=15s 480s "$PY" probe.py --output "$OSCAR_OUTPUT_DIR/probe.json"
rc=$?
set -e
printf '%s\n' "$rc" > "$OSCAR_OUTPUT_DIR/probe-exit-code"
scontrol show job -o "$SLURM_JOB_ID" > "$OSCAR_OUTPUT_DIR/scheduler-after.txt"
exit "$rc"
