# Archive packed-QKV implementation and two setup failures

Packed BF16 Q/K/V replaces three projections with one while retaining original weights and strict score checks. Neither allocation measured the candidate:8135405 exhausted its1500-second generation budget in original-arm setup;8144136 failed original-arm static-runner attestation after951.199seconds, within its deadline. Missing/multiple/incomplete runner-record cause is unknown because the in-memory records were not retained. The observer approach is parked after two feasibility interventions. This PR archives both complete130-file source snapshots, unchanged negative outcomes, and preparation evidence. There is no QKV latency, quality or exactness result and no third allocation authorized by this publication.

[Saved evidence, source snapshots and provenance](../research/packed-qkv-feasibility/README.md). Published retrospectively on2026-10-31; original review dates are retained. No adoption or merge.
