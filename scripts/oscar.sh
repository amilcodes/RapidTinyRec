#!/bin/bash
# Minimal Oscar job runner.
#   oscar.sh submit NAME PARTITION GPU TIME -- shell command (runs in code dir)
#   oscar.sh fetch NAME          # rsync results back to ./runs/NAME
#   oscar.sh status              # squeue for this user
#   oscar.sh log NAME [N]        # tail slurm log
# The command sees $PY, $REL, $REF, $OUT.
set -euo pipefail
HOST=aagraw73@sshcampus.ccv.brown.edu
ROOT=/oscar/scratch/aagraw73/speedrun
SSHO="-o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=20"
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOCAL_RUNS="$REPO/../runs"
case "$1" in
submit)
  name=$2; part=$3; gpu=$4; tlim=$5; shift 6
  run=$ROOT/runs/$name
  ssh $SSHO $HOST "test ! -e $run/out/slurm.log || { echo 'run exists: $name' >&2; exit 3; }; mkdir -p $run/code $run/out"
  rsync -az -e "ssh $SSHO" --exclude .git --exclude __pycache__ "$REPO/rapidtinyrec" "$REPO/scripts" "$HOST:$run/code/"
  qos=norm-gpu; [ "$part" = gpu-debug ] && qos=gpu-debug
  cat > "/tmp/sr-$name.sbatch" <<EOF
#!/bin/bash
#SBATCH -J sr-$name
#SBATCH -p $part
#SBATCH -q $qos
#SBATCH --gres=gpu:$gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t $tlim
#SBATCH -o $run/out/slurm.log
set -euo pipefail
export PY=/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python
export REL=/oscar/scratch/aagraw73/minionerec-shared-prefill/release
export REF=/oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users
export OUT=$run/out
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/oscar/scratch/aagraw73/speedrun/pylib
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor TRITON_CACHE_DIR=$ROOT/cache/triton
cd $run/code
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "START \$(date -u +%FT%TZ)"
$*
echo "END \$(date -u +%FT%TZ)"
EOF
  scp -q $SSHO "/tmp/sr-$name.sbatch" "$HOST:$run/job.sbatch"
  ssh $SSHO $HOST "sbatch --parsable $run/job.sbatch"
  ;;
fetch)
  name=$2; mkdir -p "$LOCAL_RUNS/$name"
  rsync -az -e "ssh $SSHO" --exclude '*.pt' "$HOST:$ROOT/runs/$name/out/" "$LOCAL_RUNS/$name/"
  echo "$LOCAL_RUNS/$name"
  ;;
status)
  ssh $SSHO $HOST "squeue -u aagraw73 -o '%.10i %.30j %.10T %.10M %.12l %.20R %.15b'"
  ;;
log)
  ssh $SSHO $HOST "tail -n ${3:-40} $ROOT/runs/$2/out/slurm.log"
  ;;
esac
