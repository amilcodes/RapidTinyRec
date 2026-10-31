"""Guard completeness and failure semantics of the preserved timeout archive."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]/'benchmarks/diagnostics/precision-pilot-timeout'


class TimeoutEvidenceTest(unittest.TestCase):
    def test_all_archived_bytes_and_submitted_sources(self):
        manifest = json.loads((ROOT/'archive-manifest.json').read_text())
        members = {}
        for archive in manifest['archives']:
            path = ROOT/archive['path']
            self.assertLess(path.stat().st_size, 48*1024*1024)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), archive['sha256'])
            with tarfile.open(path, 'r:gz') as tf:
                for member in tf.getmembers():
                    self.assertTrue(member.isfile())
                    self.assertNotIn(member.name, members)
                    data = tf.extractfile(member).read()
                    expected = manifest['files'][member.name]
                    self.assertEqual(len(data), expected['bytes'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), expected['sha256'])
                    members[member.name] = data
        self.assertEqual(set(members), set(manifest['files']))
        raw = json.loads(members['review/raw-manifest.json'])['files']
        self.assertEqual(len(raw), 235)
        self.assertEqual({n[4:] for n in members if n.startswith('raw/')}, {r['path'] for r in raw})
        for row in raw:
            self.assertEqual(hashlib.sha256(members['raw/'+row['path']]).hexdigest(), row['sha256'])
        record = json.loads(members['review/run-record.json'])
        self.assertEqual(len(record['sha256']), 59)
        for name, digest in record['sha256'].items():
            self.assertEqual(hashlib.sha256(members['source/'+name]).hexdigest(), digest)
        self.assertEqual((ROOT/'summary.json').read_bytes(), members['review/independent-review.json'])
        self.assertEqual((ROOT/'failed-audit.json').read_bytes(), members['review/failed-audit.json'])

    def test_incomplete_run_cannot_be_promoted(self):
        summary = json.loads((ROOT/'summary.json').read_text())
        self.assertEqual(summary['scheduler_state'], 'TIMEOUT')
        self.assertEqual(summary['completed_phases'], [0, 1])
        self.assertEqual(summary['absent_phases'], [3])
        for key in ('original_integrity_audit_passed', 'full_performance_gate_evaluable',
                    'eligible_for_full_quality_review', 'promotion', 'noninferiority_established'):
            self.assertIs(summary[key], False)
        failed = json.loads((ROOT/'failed-audit.json').read_text())
        self.assertIs(failed['integrity_audit_passed'], False)
        self.assertIn('phase_2/summary.json', failed['traceback'])


if __name__ == '__main__':
    unittest.main()
