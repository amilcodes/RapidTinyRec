"""Check published evidence independently of the candidate and exporter."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/full500/beam-replay'
VARIANTS = ('folded_prefix_replay', 'beam_replay')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(values):
    return b''.join(struct.pack('!f', x) for x in values)


class BeamReplayFullEvidenceTests(unittest.TestCase):
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
        self.assertEqual(len(source['submitted_source_sha256']), 49)

    def test_audit_revision_and_pilot_gate_preserved(self):
        with tarfile.open(ROOT / 'evidence.tar.gz') as archive:
            status=json.load(archive.extractfile('status.json'))
            audit=json.load(archive.extractfile('audit.json'))
            pilot=json.load(archive.extractfile('pilot-gate/status.json'))
            self.assertEqual(pilot['original_audit']['audit_exit_code'],1)
            self.assertEqual(sha(archive.extractfile('audit-source/rapidtinyrec/audit_beam_replay.py').read()),audit['audit_script_sha256'])
            self.assertEqual(sha(archive.extractfile('audit-source/rapidtinyrec/audit_beam_helpers.py').read()),audit['audit_helpers_sha256'])
            revision=json.load(archive.extractfile('audit-source/audit-source-manifest.json'))
            self.assertEqual(revision['commit'],status['audit_revision_commit'])
            for name,expected in revision['sha256'].items():
                self.assertEqual(sha(archive.extractfile('audit-source/'+name).read()),expected)
            changed=[n for n in revision['sha256'] if archive.extractfile('audit-source/'+n).read()!=archive.extractfile('source-snapshot/'+n).read()]
            self.assertEqual(set(changed),{'rapidtinyrec/audit_beam_replay.py','rapidtinyrec/audit_beam_helpers.py'})
            for name,key in [('rapidtinyrec_beam_replay_pilot_audit.json','pilot_audit_sha256'),('rapidtinyrec_beam_replay_pilot_independent_review.json','pilot_review_sha256')]:
                self.assertEqual(sha(archive.extractfile('pilot-gate/'+name).read()),status[key])
            full_source=json.loads((ROOT/'source.json').read_text())
            pilot_source=json.loads((ROOT.parents[1]/'pilots/beam-replay/source.json').read_text())
            self.assertEqual(full_source['submitted_source_sha256'],pilot_source['submitted_source_sha256'])

    def test_all_probe_categories_and_integer_outputs_preserved(self):
        with (ROOT/'helper-probes.csv').open() as file:rows=list(csv.DictReader(file))
        self.assertEqual(len(rows),5880)
        self.assertEqual({r['category'] for r in rows},{'graph_direct','direct_repeated','retained_after_five_steps'})
        self.assertEqual({r['dtype'] for r in rows},{'torch.float32','torch.int32','torch.int64','torch.bool'})
        for phase in ('1','2'):
            for category in ('graph_direct','direct_repeated','retained_after_five_steps'):
                selected=[r for r in rows if r['phase']==phase and r['category']==category]
                self.assertEqual(len(selected),980)
                self.assertEqual(sum(int(r['values']) for r in selected),1624000)
                self.assertEqual(sum(int(r['matching_bits']) for r in selected),1624000)
                self.assertTrue(all(float(r['max_abs_finite'])==0 and r['left_sha256']==r['right_sha256'] for r in selected))

    def test_raw_bits_repeats_and_exported_pairs(self):
        with (ROOT / 'requests.csv').open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 500)
        self.assertEqual(len({row['cohort_index'] for row in rows}), 500)
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

    def test_latency_strata_and_mixed_performance_preserved(self):
        summary = json.loads((ROOT / 'summary.json').read_text())
        audit = json.loads((ROOT / 'audit.json').read_text())
        for name, value in audit.items():
            self.assertEqual(summary[name], value)
        with (ROOT / 'requests.csv').open() as file:
            rows = list(csv.DictReader(file))
        groups = [(rows, summary['latency_ms'])]
        groups += [([r for r in rows if r['prompt_tokens'] == length], value)
                   for length, value in summary['latency_by_length_ms'].items()]
        groups += [([r for r in rows if int(r['cohort_index']) % 2 == int(half)], value)
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
        self.assertLess(summary['latency_ms'][VARIANTS[1]]['median_ms'],
                           summary['latency_ms'][VARIANTS[0]]['median_ms'])
        self.assertGreater(summary['latency_by_length_ms']['111']['beam_replay']['median_ms'],summary['latency_by_length_ms']['111']['folded_prefix_replay']['median_ms'])
        self.assertGreater(summary['latency_by_length_ms']['116']['beam_replay']['p95_ms'],summary['latency_by_length_ms']['116']['folded_prefix_replay']['p95_ms'])
        self.assertEqual(summary['comparison']['matching_rank_positions'],25000)
        self.assertEqual(summary['comparison']['matching_sequence_values'],25000)
        self.assertEqual(summary['comparison']['matching_transition_values'],125000)
        self.assertEqual(summary['quality']['independent_users'], 500)
        for changes in summary['quality']['paired_mean_changes'].values():
            self.assertEqual(set(changes.values()), {0.0})


if __name__ == '__main__':
    unittest.main()
