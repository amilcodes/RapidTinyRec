#!/bin/bash
# OpenOneRec certification-cost study (scripts/openonerec_cert.py).
# usage:
#   bash scripts/openonerec_job.sh download        # login node: model (ungated) + benchmark files (gated: run
#                                                  #   `huggingface-cli login` first, after accepting the dataset terms)
#   bash scripts/openonerec_job.sh inspect         # login node: columns, a sample row, catalog key format
#   bash scripts/openonerec_job.sh run TASK [N]    # GPU job: TASK in video|ad|product, N requests (default 200)
#   bash scripts/openonerec_job.sh latency TASK [N] [REF]   # GPU job: latency of certified vs beam in our engine
#     catalog: the benchmark's sid2pid.json (video/ad) or sid2iid.json (product); CATALOG=$D/RecIF/video_ad_pid2sid.parquet
#     gives the full 14.6M-SID corpus. Ground truth comes from metadata['answer'].
set -u
S=/oscar/scratch/aagraw73/speedrun
D=$S/openonerec
PYB=${PY:-/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python}
case $1 in
download)
  mkdir -p $D
  PYTHONPATH= HF_HUB_OFFLINE=0 $PYB - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('OpenOneRec/OneRec-1.7B', local_dir='$D/OneRec-1.7B', allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt', 'vocab.json'])
for f in ['benchmark_data/video/video_test.parquet', 'benchmark_data/ad/ad_test.parquet',
          'benchmark_data/product/product_test.parquet', 'benchmark_data/sid2pid.json', 'benchmark_data/sid2iid.json',
          'video_ad_pid2sid.parquet', 'product_pid2sid.parquet']:
    print(hf_hub_download('OpenOneRec/OpenOneRec-RecIF', f, repo_type='dataset', local_dir='$D/RecIF'))
EOF
  du -sh $D/*
  ;;
inspect)
  PYTHONPATH=$S/pylib_oo $PYB - <<EOF
import json, pandas as pd
for t in ['video', 'ad', 'product']:
    df = pd.read_parquet('$D/RecIF/benchmark_data/%s/%s_test.parquet' % (t, t))
    print('==', t, len(df), list(df.columns))
    r = df.iloc[0].to_dict()
    for k, v in r.items():
        print('  ', k, ':', str(v)[:700])
for f in ['sid2pid.json', 'sid2iid.json']:
    d = json.load(open('$D/RecIF/benchmark_data/' + f))
    ks = list(d)[:3] if isinstance(d, dict) else d[:3]
    print('==', f, type(d).__name__, len(d), ks, [str(d[k])[:120] for k in ks] if isinstance(d, dict) else '')
for f in ['video_ad_pid2sid.parquet', 'product_pid2sid.parquet']:
    df = pd.read_parquet('$D/RecIF/' + f)
    print('==', f, len(df), list(df.columns), df.iloc[0].to_dict())
EOF
  ;;
run)
  T=$2; N=${3:-200}
  export PYTHONPATH=$S/pylib_oo:${PYTHONPATH:-}     # pyarrow for the benchmark parquet files
  CAT=${CATALOG:-$D/RecIF/benchmark_data/sid2pid.json}
  [ "$T" = product ] && CAT=${CATALOG:-$D/RecIF/benchmark_data/sid2iid.json}
  $PY scripts/openonerec_cert.py --model $D/OneRec-1.7B --data $D/RecIF/benchmark_data/$T/${T}_test.parquet \
    --catalog $CAT --out $OUT/${T}.jsonl --users $N --target-field metadata ${EXTRA:-} 2>&1 \
    | grep --line-buffered -E '^\{"stage"|Error|Traceback|error' | cut -c1-1500
  ;;
latency)                                          # GPU job: certified vs beam latency in our engine (Qwen3 port)
  T=$2; N=${3:-200}; R=${4:-$T}                   # R: reference run dir name under runs/ (oo_$R/$T.jsonl)
  export PYTHONPATH=$S/pylib_oo:${PYTHONPATH:-}
  CAT=${CATALOG:-$D/RecIF/benchmark_data/sid2pid.json}
  [ "$T" = product ] && CAT=${CATALOG:-$D/RecIF/benchmark_data/sid2iid.json}
  $PY scripts/openonerec_latency.py --model $D/OneRec-1.7B --data $D/RecIF/benchmark_data/$T/${T}_test.parquet \
    --catalog $CAT --out $OUT/${R}.jsonl --users $N --ref $S/runs/oo_$R/out/${T}.jsonl ${EXTRA:-} 2>&1 \
    | grep --line-buffered -E '^\{"stage"|Error|Traceback|error|line [0-9]' | cut -c1-2500
  ;;
smoke)
  export PYTHONPATH=$S/pylib_oo:${PYTHONPATH:-}
  $PY scripts/openonerec_cert.py --model $D/OneRec-1.7B --data $D/smoke/smoke_test.parquet --catalog $D/smoke/catalog.json \
    --out $OUT/smoke.jsonl --users 4 --ks 1,10,32 --beams 32 --check 2 2>&1 | tail -40
  head -c 3000 $OUT/smoke.jsonl
  ;;
esac
