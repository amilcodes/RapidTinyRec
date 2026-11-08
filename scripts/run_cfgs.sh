#!/bin/bash
# usage: run_cfgs.sh NAME1 "args1" NAME2 "args2" ...   (continues past failures; retries CUDA-init failures)
set -u
while [ $# -ge 2 ]; do
  name=$1; args=$2; shift 2
  for attempt in 1 2 3; do
    echo "=== CONFIG $name (attempt $attempt): $args  $(date -u +%T)"
    log=$OUT/$name.log
    mkdir -p $OUT
    $PY scripts/bench.py --release $REL --ref $REF --out $OUT/$name $args > $log 2>&1
    rc=$?
    grep --line-buffered -v -E "SettingWithCopy|it/s\]|caveats|history_item_sid|^$|A value is trying" $log | grep -E '^\{"stage"|Error|error|Traceback' | cut -c1-3000
    echo "=== EXIT $name $rc  $(date -u +%T)"
    if [ $rc -ne 0 ] && grep -q "CUDA driver initialization failed\|CUDA-capable device\|previous error during capture" $log; then sleep 20; continue; fi
    break
  done
done
