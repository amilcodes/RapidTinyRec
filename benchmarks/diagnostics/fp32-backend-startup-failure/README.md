# FP32 backend startup failure

Job7955758 failed before model loading or any projection measurements. This is an implementation/provenance failure, not a measured numerical or performance failure. The installed PyTorch release wheel is the immediate parent of the reviewed release tag; the tag adds only a test-file import. All five reviewed numerical/backend source files are byte-identical. The corrected diagnostic pins the observed wheel commit; exactness gates remain unchanged.

`summary.json` records the failure and original audit command. Extract every `evidence-*.tar.gz` into one folder; `archive-manifest.json` hashes every member and archive. The archive preserves all raw outputs/logs, original source/config/resources, run record, CI log, original failed audit, and official upstream build-identity evidence. It also includes the publication and guarded-submission utilities.

To reproduce the original failed audit, from extracted `source-snapshot` run:

```bash
python3 -m rapidtinyrec.audit_fp32_backend ../raw/outputs/diagnosis --snapshot . --run-record ../run-record.json --output ../reproduced-failure-audit.json
```

Expected exit1 with the preserved missing-release-provenance error. No score, ranking or latency claim follows from this run. Do not submit the archived job again.
