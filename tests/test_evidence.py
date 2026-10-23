"""Check published measurements without downloading a model or importing torch."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_every_measured_request_is_accounted_for(self):
        paths = list((ROOT/'benchmarks/results').glob('*/summary.json'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(experiment=path.parent.name):
                summary = json.loads(path.read_text())
                csv_path = path.with_name('requests.csv')
                self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(), summary['requests_sha256'])
                with csv_path.open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([int(r['cohort_index']) for r in rows], list(range(500)))
                control, candidate = summary['control'], summary['candidate']
                for variant, expected in summary['latency_ms'].items():
                    values = [float(r[variant+'_ms']) for r in rows]
                    self.assertTrue(all(v > 0 for v in values))
                    self.assertAlmostEqual(statistics.median(values), expected['median'], places=9)
                    self.assertAlmostEqual(statistics.quantiles(values,n=100,method='inclusive')[94], expected['p95'], places=9)
                for key in ['matching_rank_positions','matching_sequence_values','matching_transition_values']:
                    self.assertEqual(sum(int(r[key]) for r in rows), summary[key])
                self.assertEqual(sum(int(r['matching_rank_positions'])==50 for r in rows),summary['matching_rank_lists'])
                for r in rows:
                    self.assertEqual(r[control+'_ranking_sha256']==r[candidate+'_ranking_sha256'],int(r['matching_rank_positions'])==50)
                    for field, count, maximum in [('sequence_logp','sequence',50),('transition_logp','transition',250)]:
                        self.assertEqual(r[control+'_'+field+'_sha256']==r[candidate+'_'+field+'_sha256'],int(r['matching_'+count+'_values'])==maximum)
                for label in ['sequence','transition']:
                    self.assertEqual(max(float(r['max_'+label+'_difference']) for r in rows),summary['max_'+label+'_difference'])

    def test_pinned_reference_sources(self):
        expected=json.loads((ROOT/'rapidtinyrec/expected.json').read_text())
        for name, digest in expected['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'rapidtinyrec'/name).read_bytes()).hexdigest(),digest,name)


if __name__ == '__main__':
    unittest.main()
