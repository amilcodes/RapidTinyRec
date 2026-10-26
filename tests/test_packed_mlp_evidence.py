"""Recompute the rejected pilot from archived outputs without running a model."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'benchmarks/pilots/packed-mlp'
VARIANTS = ('folded_prefix_replay', 'packed_mlp_replay')


def fp32(values):
    return [struct.pack('!f', x) for x in values]


def values(run, field):
    return (run[field] if field == 'sequence_logp'
            else [x for row in run[field] for x in row])


class PackedMlpEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((EVIDENCE / 'audit.json').read_text())
        cls.summary = json.loads((EVIDENCE / 'summary.json').read_text())
        cls.source = json.loads((EVIDENCE / 'source.json').read_text())
        with tarfile.open(EVIDENCE / 'evidence.tar.gz') as archive:
            members = archive.getmembers()
            assert all(m.isfile() for m in members)
            assert len({m.name for m in members}) == len(members)
            cls.files = {m.name: archive.extractfile(m).read() for m in members}

    def read(self, name):
        return json.loads(self.files[name])

    def compare(self, pairs, expected):
        self.assertEqual(len(pairs), expected['requests'])
        self.assertEqual(sum(a['token_ids'] == b['token_ids'] for a, b in pairs),
                         expected['matching_rank_lists'])
        self.assertEqual(sum(x == y for a, b in pairs for x, y in zip(a['token_ids'], b['token_ids'])),
                         expected['matching_rank_positions'])
        for field, label, vectors in [('sequence_logp', 'sequence', 'vectors'),
                                      ('transition_logp', 'transition', 'arrays')]:
            arrays = [(values(a, field), values(b, field)) for a, b in pairs]
            self.assertEqual(sum(sum(x == y for x, y in zip(fp32(a), fp32(b))) for a, b in arrays),
                             expected['matching_' + label + '_values'])
            self.assertEqual(sum(fp32(a) == fp32(b) for a, b in arrays),
                             expected['matching_' + label + '_' + vectors])
            # Rankings are all identical in this rejected pilot, so positions
            # identify the same candidate when recomputing maximum deltas.
            self.assertTrue(all(a['token_ids'] == b['token_ids'] for a, b in pairs))
            self.assertEqual(max(abs(x - y) for a, b in arrays for x, y in zip(a, b)),
                             expected['max_' + label + '_difference_common'])

    def test_archive_and_source_integrity(self):
        manifest = json.loads((EVIDENCE / 'archive-manifest.json').read_text())
        self.assertEqual(set(manifest), set(self.files))
        self.assertEqual(len(manifest), 228)
        for name, expected in manifest.items():
            self.assertEqual(len(self.files[name]), expected['bytes'], name)
            self.assertEqual(hashlib.sha256(self.files[name]).hexdigest(), expected['sha256'], name)
        for name, digest in self.source['evidence_sha256'].items():
            self.assertEqual(hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(), digest, name)
        record = self.read('provenance/run-record.json')
        self.assertEqual(record['job_id'], '7907979')
        self.assertEqual(len(record['sha256']), 56)
        self.assertEqual(record['sha256'], self.source['submitted_source_sha256'])
        self.assertEqual(record['sha256'], self.read('raw/source-manifest.json')['sha256'])
        for name, digest in record['sha256'].items():
            self.assertEqual(hashlib.sha256(self.files['snapshot/' + name]).hexdigest(), digest, name)
        for name, digest in self.source['published_source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(self.read('provenance/audit.json'), self.audit)

    def test_negative_exactness_and_frozen_execution(self):
        self.assertTrue(self.audit['integrity_audit_passed'])
        self.assertFalse(self.audit['exactness_passed'])
        self.assertFalse(self.summary['exactness_passed'])
        joined = {}
        historical = {i: self.read(f'historical/users/{i:04d}.json')['runs']['compiled_gqa']
                      for i in (1, 2, 6, 10)}
        for phase in range(4):
            prefix = f'raw/outputs/phase_{phase}/'
            rows = self.read(prefix + 'requests.json')
            setup = self.read(prefix + 'setup.json')
            summary = self.read(prefix + 'summary.json')
            self.assertTrue(summary['complete'])
            self.assertEqual(len(rows), 12)
            self.assertEqual(self.read(prefix + 'compilation.json')['measurement_compilations'], 0)
            self.assertEqual(setup['replay']['graphs'], 8)
            self.assertEqual(summary['replay']['graphs'], 8)
            self.assertTrue(setup['replay']['frozen'] and summary['replay']['frozen'])
            self.assertEqual(setup['replay']['events'], summary['replay']['events'])
            for row in rows:
                key = row['repeat'], row['cohort_index']
                joined.setdefault(key, {})[row['variant']] = row['run']
            variant = rows[0]['variant']
            warm_pairs, historical_pairs = [], []
            for index in historical:
                warm = [self.read(prefix + f'warmup_outputs/{index}_{suffix}.json')
                        for suffix in ('dense_forward', variant, 'repeat')]
                warm_pairs.append((warm[0], warm[1]))
                historical_pairs.extend((historical[index], run) for run in warm)
                for field in ('sequence_logp', 'transition_logp'):
                    self.assertEqual(fp32(values(warm[1], field)), fp32(values(warm[2], field)))
            self.compare(warm_pairs, self.audit['phases'][phase]['warmup_comparison'])
            self.compare(historical_pairs, self.audit['phases'][phase]['warmup_historical_comparison'])
        self.assertEqual(len(joined), 24)
        self.compare([(r[VARIANTS[0]], r[VARIANTS[1]]) for r in joined.values()], self.audit['comparison'])
        for variant in VARIANTS:
            self.compare([(historical[index], runs[variant]) for (_, index), runs in joined.items()],
                         self.audit['historical_comparisons'][variant])
            for index in historical:
                repeats = [joined[(repeat, index)][variant] for repeat in range(6)]
                for field in ('sequence_logp', 'transition_logp'):
                    self.assertTrue(all(fp32(values(repeats[0], field)) == fp32(values(r, field)) for r in repeats))
        self.assertEqual(self.audit['comparison']['matching_sequence_values'], 78)
        self.assertEqual(self.audit['comparison']['matching_transition_values'], 1524)

    def test_csv_latency_and_mismatch_counts(self):
        with (EVIDENCE / 'requests.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        for field in ('rank_positions', 'sequence_values', 'transition_values'):
            self.assertEqual(sum(int(r['matching_' + field]) for r in rows), self.summary['matching_' + field])
            self.assertEqual(self.summary['matching_' + field], self.audit['comparison']['matching_' + field])
        for label in ('sequence', 'transition'):
            self.assertEqual(max(float(r['max_' + label + '_difference']) for r in rows),
                             self.summary['max_' + label + '_difference'])
        for variant in VARIANTS:
            for group, expected in [(rows, self.audit['latency_ms'][variant])]:
                self.check_latency(group, variant, expected)
            for length, expected in self.audit['latency_by_length_ms'].items():
                self.check_latency([r for r in rows if r['prompt_tokens'] == length], variant, expected[variant])
            for half, expected in self.audit['latency_by_half_ms'].items():
                self.check_latency([r for r in rows if int(r['repeat']) // 3 == int(half)], variant, expected[variant])
            for row in rows:
                phase = row[variant + '_phase']
                raw = self.read(f"raw/outputs/phase_{phase}/users/{row['repeat']}_{int(row['cohort_index']):04d}.json")['run']
                self.assertEqual(float(row[variant + '_ms']), raw['elapsed_s'] * 1000)

    def check_latency(self, rows, variant, expected):
        durations = [float(r[variant + '_ms']) for r in rows]
        self.assertEqual(len(durations), expected['n'])
        self.assertEqual(statistics.median(durations), expected['median_ms'])
        self.assertEqual(statistics.quantiles(durations, n=100, method='inclusive')[94], expected['p95_ms'])


if __name__ == '__main__':
    unittest.main()
