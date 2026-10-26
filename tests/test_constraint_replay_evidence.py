"""Check published evidence independently of the candidate and exporter."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/pilots/constraint-replay'
VARIANTS = ('folded_prefix_replay', 'constraint_replay')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(values):
    return b''.join(struct.pack('!f', x) for x in values)


class ConstraintReplayEvidenceTests(unittest.TestCase):
    def test_archive_and_submitted_source_integrity(self):
        source = json.loads((ROOT / 'source.json').read_text())
        manifest = json.loads((ROOT / 'archive-manifest.json').read_text())
        for name, expected in source['evidence_sha256'].items():
            self.assertEqual(sha((ROOT / name).read_bytes()), expected, name)
        with tarfile.open(ROOT / 'evidence.tar.gz') as archive:
            self.assertEqual(len(archive.getmembers()), len(manifest))
            self.assertEqual(set(archive.getnames()), set(manifest))
            for item in archive:
                self.assertTrue(item.isfile())
                data = archive.extractfile(item).read()
                self.assertEqual(len(data), manifest[item.name]['bytes'])
                self.assertEqual(sha(data), manifest[item.name]['sha256'])
            for name, expected in source['submitted_source_sha256'].items():
                self.assertEqual(sha(archive.extractfile('source-snapshot/' + name).read()), expected)
            record = json.load(archive.extractfile('run-record.json'))
            self.assertEqual(record['sha256'], source['submitted_source_sha256'])
            self.assertEqual(record['resources'], source['resources'])
            self.assertEqual(record['job_id'], source['job_id'])
        self.assertEqual(len(source['submitted_source_sha256']), 48)

    def test_raw_bits_repeats_and_exported_pairs(self):
        with (ROOT / 'requests.csv').open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row['cohort_index'] for row in rows}), 4)
        repeated = {}
        with tarfile.open(ROOT / 'evidence.tar.gz') as archive:
            for row in rows:
                outputs = []
                for variant in VARIANTS:
                    phase = row[variant + '_phase']
                    name = f"raw/outputs/phase_{phase}/users/{row['repeat']}_{int(row['cohort_index']):04d}.json"
                    run = json.load(archive.extractfile(name))['run']
                    self.assertEqual(run['elapsed_s'] * 1000, float(row[variant + '_ms']))
                    values = (json.dumps(run['token_ids'], separators=(',', ':')).encode(),
                              bits(run['sequence_logp']),
                              bits([x for scores in run['transition_logp'] for x in scores]))
                    for field, value in zip(('ranking', 'sequence_logp', 'transition_logp'), values):
                        self.assertEqual(sha(value), row[variant + '_' + field + '_sha256'])
                    key = variant, row['cohort_index']
                    self.assertEqual(repeated.setdefault(key, values), values)
                    outputs.append(values)
                self.assertEqual(*outputs)
                self.assertEqual(int(row['matching_rank_positions']), 50)
                self.assertEqual(int(row['matching_sequence_values']), 50)
                self.assertEqual(int(row['matching_transition_values']), 250)

    def test_latency_strata_and_negative_promotion_preserved(self):
        summary = json.loads((ROOT / 'summary.json').read_text())
        audit = json.loads((ROOT / 'audit.json').read_text())
        for name, value in audit.items():
            self.assertEqual(summary[name], value)
        with (ROOT / 'requests.csv').open() as file:
            rows = list(csv.DictReader(file))
        groups = [(rows, summary['latency_ms'])]
        groups += [([r for r in rows if r['prompt_tokens'] == length], value)
                   for length, value in summary['latency_by_length_ms'].items()]
        groups += [([r for r in rows if int(r['repeat']) // 3 == int(half)], value)
                   for half, value in summary['latency_by_half_ms'].items()]
        for group, expected in groups:
            for variant in VARIANTS:
                values = [float(r[variant + '_ms']) for r in group]
                self.assertEqual(len(values), expected[variant]['n'])
                self.assertEqual(statistics.median(values), expected[variant]['median_ms'])
                self.assertAlmostEqual(statistics.quantiles(values, n=100, method='inclusive')[94],
                                       expected[variant]['p95_ms'], places=10)
        self.assertTrue(summary['integrity_audit_passed'])
        self.assertTrue(summary['exactness_passed'])
        self.assertFalse(summary['performance_gate'])
        self.assertGreater(summary['latency_ms'][VARIANTS[1]]['median_ms'],
                           summary['latency_ms'][VARIANTS[0]]['median_ms'])
        self.assertEqual(summary['quality']['independent_users'], 4)
        for changes in summary['quality']['paired_mean_changes'].values():
            self.assertEqual(set(changes.values()), {0.0})


if __name__ == '__main__':
    unittest.main()
