#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
test -n "${OSCAR_RUN_DIR:-}"
python_bin=/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python
if (( $# == 0 )); then
    exec "$python_bin" -u -m rapidtinyrec.qkv_pilot_runner run
fi
test "$#" -eq 2
test "$1" = --child
test "$2" = "${RAPIDTINYREC_LOG_STAGE_WRAPPER:-}"
test -d "$2"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCH_LOGS=recompiles
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_TUNABLEOP_TUNING=0
for phase in 0 1 2 3; do
    if (( phase == 0 || phase == 3 )); then variant=original_bf16; else variant=packed_qkv_bf16; fi
    export TORCHINDUCTOR_CACHE_DIR="$OSCAR_RUN_DIR/compile-cache/$variant/inductor"
    export TRITON_CACHE_DIR="$OSCAR_RUN_DIR/compile-cache/$variant/triton"
    if (( phase < 2 )); then
        test ! -e "$TORCHINDUCTOR_CACHE_DIR"
        test ! -e "$TRITON_CACHE_DIR"
    else
        test -d "$TORCHINDUCTOR_CACHE_DIR"
        test -d "$TRITON_CACHE_DIR"
    fi
    "$python_bin" -u -m rapidtinyrec.qkv_phase_entry --phase "$phase" --api-log-local-root "$2"
done
