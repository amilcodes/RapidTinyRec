#!/bin/bash
# One-time: FlashRec 0.1.2 (sohu-mptc/FlashRec @ 1089682, Apache-2.0) in its own venv on Oscar scratch, with the
# Qwen2 patch (scripts/flashrec_qwen2.py). Run on a login node from a synced copy of this repo:
#   bash scripts/flashrec_setup.sh
set -euo pipefail
S=/oscar/scratch/aagraw73/speedrun
BASEPY=/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python
CODE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $S/src $S/envs
if [ ! -d $S/src/FlashRec ]; then git clone -q https://github.com/sohu-mptc/FlashRec.git $S/src/FlashRec; fi
cd $S/src/FlashRec
git checkout -q 1089682
git checkout -q -- python                 # idempotent: re-apply the patch on a clean tree
$BASEPY $CODE/scripts/flashrec_qwen2.py $S/src/FlashRec
git diff --stat | cat
if [ ! -x $S/envs/flashrec/bin/python ]; then $BASEPY -m venv $S/envs/flashrec; fi
FP=$S/envs/flashrec/bin/python
$FP -m pip install -q --upgrade pip
# FlashRec pins torch>=2.9,<3, sgl-kernel>=0.3.21,<0.4, flashinfer 0.6.x, triton>=3.5; CUDA 12.8 wheels
$FP -m pip install -q "torch>=2.9,<2.10" --index-url https://download.pytorch.org/whl/cu128
$FP -m pip install -q -e "$S/src/FlashRec"
$FP -m pip list 2>/dev/null | grep -i -E "^(torch|sgl-kernel|flashinfer|triton|transformers|apache-tvm) " | cat
$FP -c "import flashrec, torch; print('flashrec', flashrec.__version__, 'torch', torch.__version__, torch.version.cuda)"
