# Archive buffer-reuse pilot and incomplete inventory finding

The buffer-reuse pilot8120807 preserved the recorded five full-score hashes, tokens, sequence scores and ancestry, but missed the performance gate: median85.3308→84.7885ms(-0.6355%), p9586.7430→87.1876ms(+0.5126%). None of nine strata reaches5% median reduction. Source review additionally found five nested tensors missing from the immutable inventory. Complete protocol certification and full500 eligibility remain false despite the raw execution flag. This PR archives the exact99-file implementation and its rejection, without integrating it into serving.

## Chronology and scope

Work/evidence date: 2026-10-30T07:59:18.044877+00:00. Published retrospectively on2026-10-31. When an exact review timestamp is available, the Git author date records that timestamp; committer and PR creation/update dates are current. The TF32 recovery publication uses a current author date because no precise review timestamp is present.

The files under `evidence/` are byte-for-byte copies of saved reviews, manifests and reconciliation records. Historical paths and next-action text inside them are provenance from their original date, not current instructions. The narrative above is the publication decision. `manifest.json` records the original workspace-relative location, byte count and SHA256 for each copied artifact.

Source snapshots, when included, preserve the exact submitted bytes and source locks. They are historical experiment packages with cluster-specific paths, not installed serving modules. Do not execute or resubmit closed runs or reuse their readiness locks. No serving defaults, scoring objective, precision policy, production integration, or merge is changed here.

This is a curated evidence publication, not a complete raw-data archive. Full raw tensor/output/trace collections remain in the frozen local experiment archive and are not uploaded; manifests and saved review claims do not substitute for those raw files. Any paths outside this folder identify that retained archive and are not portable links. Negative strata and uncertainty are retained in full JSON reviews, rather than replaced by headline aggregates.

## Validation

Packaging verifies every copied byte against its original file, and each source snapshot against its existing reviewed lock. JSON parsing and Python syntax checks do not import experiment code or repeat numerical tests. Prior component test results remain historical evidence; no GPU job, numerical re-audit, old CI rerun or speed measurement is performed for this publication. `[skip ci]` prevents replaying unchanged historical suites for this artifact-only update.

From this folder, `python3 verify_publication.py` verifies the exact published file inventory, sizes, hashes and JSON syntax without Torch, model execution or network access. It does not establish scientific correctness independently of the recorded audits.

Snapshot file counts: {"submitted": 99}.
