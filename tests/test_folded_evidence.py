"""Recompute published folded-attention evidence without a GPU or private data."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FoldedEvidenceTests(unittest.TestCase):
    def test_pilot_pairs_exactness_and_latency(self):
        self._check_pairs(ROOT / 'benchmarks/pilots/folded-query-heads', 24, 4, 6)

    def test_full_pairs_exactness_and_latency(self):
        self._check_pairs(ROOT / 'benchmarks/results/folded-query-heads', 500, 500, 1)

    def _check_pairs(self, root, requests, users, repeats):
        summary = json.loads((root / 'summary.json').read_text())
        audit = json.loads((root / 'audit.json').read_text())
        with (root / 'requests.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), requests)
        self.assertEqual(len({r['cohort_index'] for r in rows}), users)
        self.assertEqual(summary['paired_requests'], requests)
        self.assertEqual(summary['users'], users)
        for index in {r['cohort_index'] for r in rows}:
            group = [r for r in rows if r['cohort_index'] == index]
            self.assertEqual({r['repeat'] for r in group}, {str(i) for i in range(repeats)})
        if users == 500:
            self.assertEqual([int(r['cohort_index']) for r in rows], list(range(500)))
            for row in rows:
                phases = ('0', '1') if int(row['cohort_index']) % 2 == 0 else ('3', '2')
                self.assertEqual((row['prefix_replay_phase'], row['folded_prefix_replay_phase']), phases)
        self.assertTrue(audit['integrity_audit_passed'])
        self.assertTrue(audit['exactness_passed'])
        self.assertEqual(summary['exactness_passed'], audit['exactness_passed'])
        for variant, expected in summary['latency_ms'].items():
            values = [float(r[variant + '_ms']) for r in rows]
            self.assertTrue(all(v > 0 for v in values))
            self.assertEqual(statistics.median(values), expected['median'])
            self.assertEqual(statistics.quantiles(values, n=100, method='inclusive')[94], expected['p95'])
        left, right = summary['control'], summary['candidate']
        comparisons = [audit['comparison'], *audit['historical_comparisons'].values()]
        for field, count in [('rank_positions', requests * 50), ('sequence_values', requests * 50), ('transition_values', requests * 250)]:
            total = sum(int(row['matching_' + field]) for row in rows)
            self.assertEqual(total, count)
            self.assertEqual(total, summary['matching_' + field])
            for comparison in comparisons:
                self.assertEqual(comparison['matching_' + field], count)
                self.assertEqual(comparison[field], count)
        for row in rows:
            for field in ('ranking', 'sequence_logp', 'transition_logp'):
                self.assertEqual(row[left + '_' + field + '_sha256'], row[right + '_' + field + '_sha256'])
        for field in ('sequence', 'transition'):
            self.assertEqual(max(float(r['max_' + field + '_difference']) for r in rows), 0)
            self.assertEqual(summary['max_' + field + '_difference'], 0)
        for phase in audit['phases']:
            self.assertEqual(phase['measurement_compilations'], 0)
            self.assertEqual(phase['measurement_captures'], 0)
            self.assertEqual(phase['compiler_artifacts_reused'], phase['phase'] >= 2)
            for key in ('warmup_comparison', 'warmup_historical_comparison'):
                for field in ('rank_positions', 'sequence_values', 'transition_values'):
                    self.assertEqual(phase[key][field], phase[key]['matching_' + field])

    def test_evidence_and_source_hashes(self):
        for category in ('pilots', 'results'):
            with self.subTest(category=category):
                self._check_hashes(ROOT / 'benchmarks' / category / 'folded-query-heads')

    def _check_hashes(self, root):
        source = json.loads((root / 'source.json').read_text())
        summary = json.loads((root / 'summary.json').read_text())
        self.assertEqual(hashlib.sha256((root / 'requests.csv').read_bytes()).hexdigest(), summary['requests_sha256'])
        self.assertEqual(len(source['submitted_source_sha256']), 49)
        for name, expected in source['evidence_sha256'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)
        for name, expected in source['published_source_sha256'].items():
            self.assertEqual(source['submitted_source_sha256'][name], expected, name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)


if __name__ == '__main__':
    unittest.main()
