# Archive buffer-reuse pilot and incomplete inventory finding

The buffer-reuse pilot8120807 preserved the recorded five full-score hashes, tokens, sequence scores and ancestry, but missed the performance gate: median85.3308→84.7885ms(-0.6355%), p9586.7430→87.1876ms(+0.5126%). None of nine strata reaches5% median reduction. Source review additionally found five nested tensors missing from the immutable inventory. Complete protocol certification and full500 eligibility remain false despite the raw execution flag. This PR archives the exact99-file implementation and its rejection, without integrating it into serving.

[Saved evidence, source snapshots and provenance](../research/buffer-reuse-results/README.md). Publication on2026-10-31; historical work dates are retained. No adoption or automatic merge.
