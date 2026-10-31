#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
test -n "${OSCAR_RUN_DIR:-}"
test -d "${OSCAR_OUTPUT_DIR:-}"
python_bin=/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python
if (( $# == 0 )); then
    exec "$python_bin" -u -m rapidtinyrec.precision_full500_runner run \
        --run-root "$OSCAR_RUN_DIR" --local-root /tmp \
        --release-root /oscar/scratch/aagraw73/minionerec-shared-prefill/release \
        --historical /oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users
fi
test "$#" -eq 2
test "$1" = --child
test "$2" = "${RAPIDTINYREC_LOG_STAGE_WRAPPER:-}"
test -d "$2"
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_LOGS=recompiles
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_TUNABLEOP_TUNING=0
for phase in 0 1 2 3; do
    if (( phase == 0 || phase == 3 )); then variant=fp32; else variant=tf32; fi
    export TORCHINDUCTOR_CACHE_DIR="$OSCAR_RUN_DIR/compile-cache/$variant/inductor"
    export TRITON_CACHE_DIR="$OSCAR_RUN_DIR/compile-cache/$variant/triton"
    extra=()
    if (( phase < 2 )); then
        test ! -e "$TORCHINDUCTOR_CACHE_DIR"
        test ! -e "$TRITON_CACHE_DIR"
    else
        test -d "$TORCHINDUCTOR_CACHE_DIR"
        test -d "$TRITON_CACHE_DIR"
        extra+=(--reuse-compiler-artifacts)
    fi
    "$python_bin" -u -m rapidtinyrec.benchmark_precision \
        --release-root /oscar/scratch/aagraw73/minionerec-shared-prefill/release \
        --reference-results /oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users \
        --output-root "$OSCAR_OUTPUT_DIR" --phase "$phase" --api-log-local-root "$2" \
        --full-cohort --full500-quality "${extra[@]}"
done
