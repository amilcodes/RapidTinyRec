#!/bin/bash
# usage: run_cfgs.sh NAME1 "args1" NAME2 "args2" ...   (continues past failures)
set -u
while [ $# -ge 2 ]; do
  name=$1; args=$2; shift 2
  echo "=== CONFIG $name: $args  $(date -u +%T)"
  $PY scripts/bench.py --release $REL --ref $REF --out $OUT/$name $args 2>&1 | grep --line-buffered -v -E "SettingWithCopy|it/s\]|caveats|history_item_sid|^$|A value is trying"
  echo "=== EXIT $name ${PIPESTATUS[0]}  $(date -u +%T)"
done
