import csv,hashlib,json,statistics,unittest
from pathlib import Path


class FoldedProfileEvidenceTests(unittest.TestCase):
    def test_published_profile_counts_and_latency(self):
        root=Path(__file__).resolve().parents[1]/'benchmarks/profiles/folded-query'
        summary=json.loads((root/'summary.json').read_text());audit=json.loads((root/'audit.json').read_text())
        self.assertEqual(hashlib.sha256((root/'requests.csv').read_bytes()).hexdigest(),summary['requests_sha256'])
        with (root/'requests.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),60);self.assertEqual(len({r['cohort_index'] for r in rows}),10)
        values=[float(r['elapsed_ms']) for r in rows]
        self.assertEqual(statistics.median(values),summary['latency_ms']['median'])
        self.assertEqual(statistics.quantiles(values,n=100,method='inclusive')[94],summary['latency_ms']['p95'])
        self.assertTrue(audit['integrity_audit_passed'] and audit['exactness_passed'])
        for comparison in audit['historical_comparisons'].values():
            for prefix in ('rank_positions','sequence_values','transition_values'):
                self.assertEqual(comparison['matching_'+prefix],comparison[prefix])
        self.assertEqual(sum(c['requests'] for c in audit['historical_comparisons'].values()),136)
        self.assertEqual(audit['measurement_compilations'],0)
        self.assertEqual(audit['measurement_captures'],0)
        self.assertEqual(len(audit['matmul_attribution']),6)
        for index in {r['cohort_index'] for r in rows}:
            group=[r for r in rows if r['cohort_index']==index]
            self.assertEqual({r['repeat'] for r in group},{str(i) for i in range(6)})
            for field in ('ranking_sha256','sequence_fp32_sha256','transition_fp32_sha256'):
                self.assertEqual(len({r[field] for r in group}),1)


if __name__=='__main__':unittest.main()
