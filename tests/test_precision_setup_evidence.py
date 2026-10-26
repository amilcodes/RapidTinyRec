"""Guard completeness and failure semantics of the preserved setup diagnostic archive."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]/'benchmarks/diagnostics/precision-setup-feasibility'


class SetupEvidenceTest(unittest.TestCase):
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
        self.assertEqual(len(raw), 227)
        self.assertEqual({n[4:] for n in members if n.startswith('raw/')}, {r['path'] for r in raw})
        for row in raw:
            self.assertEqual(hashlib.sha256(members['raw/'+row['path']]).hexdigest(), row['sha256'])
        record = json.loads(members['review/run-record.json'])
        self.assertEqual(len(record['sha256']), 69)
        for name, digest in record['sha256'].items():
            self.assertEqual(hashlib.sha256(members['source/'+name]).hexdigest(), digest)
        self.assertEqual((ROOT/'summary.json').read_bytes(), members['review/independent-review.json'])
        self.assertEqual((ROOT/'audit.json').read_bytes(), members['review/audit.json'])

    def test_setup_pass_does_not_promote_quality(self):
        summary = json.loads((ROOT/'summary.json').read_text())
        audit = json.loads((ROOT/'audit.json').read_text())
        self.assertEqual(summary['scheduler_state'], 'COMPLETED')
        self.assertTrue(audit['integrity_audit_passed'])
        self.assertTrue(audit['setup_diagnostic_passed'])
        self.assertEqual([p['phase'] for p in audit['phases']], [1, 2])
        self.assertEqual(audit['requests'], 24)
        self.assertEqual(audit['unique_users'], 4)
        self.assertEqual(hashlib.sha256((ROOT/'audit.json').read_bytes()).hexdigest(), summary['audit_sha256'])
        for key in ('eligible_for_full_quality_review', 'promotion', 'noninferiority_established',
                    'full_abba_retry_authorized_by_diagnostic_alone'):
            self.assertIs(summary[key], False)
            self.assertIs(audit[key], False)
        self.assertIs(summary['logging_causality_established'], False)
        self.assertIs(summary['paired_performance_gate_evaluated'], False)


if __name__ == '__main__':
    unittest.main()
