"""The build-pin repair must retain the original failed run and its proof."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

from rapidtinyrec import audit_fp32_backend, fp32_backend_probe

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/diagnostics/fp32-backend-startup-failure'


class BuildEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/'archive-manifest.json').read_text())
        cls.summary=json.loads((ROOT/'summary.json').read_text())
        cls.files={}
        for shard in cls.manifest['archives']:
            data=(ROOT/shard['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==shard['sha256']
            with tarfile.open(ROOT/shard['path'],'r:gz') as archive:
                for member in archive.getmembers():
                    assert member.isfile() and member.name not in cls.files
                    cls.files[member.name]=archive.extractfile(member).read()

    def test_every_archived_byte_and_original_source_hash(self):
        self.assertEqual(set(self.files),set(self.manifest['files']))
        for name,data in self.files.items():
            self.assertEqual(len(data),self.manifest['files'][name]['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(),self.manifest['files'][name]['sha256'])
        record=json.loads(self.files['run-record.json'])
        self.assertEqual(len(record['sha256']),47)
        for name,digest in record['sha256'].items():
            self.assertEqual(hashlib.sha256(self.files['source-snapshot/'+name]).hexdigest(),digest)

    def test_failed_startup_is_not_score_or_latency_evidence(self):
        d=json.loads(self.files['raw/outputs/diagnosis/diagnosis.json'])
        self.assertFalse(d['complete']);self.assertEqual(d['cases'],[])
        self.assertEqual(d['setup'],{});self.assertEqual(d['runtime_precision'],{})
        self.assertNotIn('model_load_s',d)
        self.assertIn('Review installed build',d['failure'])
        audit=json.loads(self.files['audit.json'])
        self.assertFalse(audit['integrity_audit_passed'])
        self.assertEqual(audit['errors'],["KeyError: 'release_sha256'"])
        for flag in ('backend_exactness_established','backend_latency_measured','ranking_quality_measured'):
            self.assertFalse(self.summary[flag])

    def test_official_difference_is_test_only_and_dispatch_bytes_match(self):
        proof=json.loads(self.files['build-identity/proof.json'])
        comparison=json.loads(self.files['build-identity/github-compare.json'])
        self.assertEqual(comparison['ahead_by'],1);self.assertEqual(comparison['behind_by'],0)
        self.assertEqual([f['filename'] for f in comparison['files']],['test/test_transformers.py'])
        tag=json.loads(self.files['build-identity/reviewed-tag-source-manifest.json'])
        self.assertEqual(len(proof['source_files']),5)
        for file in proof['source_files']:
            data=self.files['build-identity/installed-source/'+file['path']]
            self.assertEqual(hashlib.sha256(data).hexdigest(),file['installed_sha256'])
            self.assertEqual(file['installed_sha256'],tag[file['path']]['sha256'])

    def test_corrected_pin_uses_observed_wheel_and_keeps_failed_guard(self):
        d=json.loads(self.files['raw/outputs/diagnosis/diagnosis.json'])
        self.assertEqual(fp32_backend_probe.TORCH_COMMIT,d['torch_git_version'])
        self.assertEqual(audit_fp32_backend.COMMIT,d['torch_git_version'])
        original=self.files['source-snapshot/rapidtinyrec/fp32_backend_probe.py'].decode()
        self.assertIn(self.summary['original_guard_commit'],original)
        self.assertNotEqual(self.summary['original_guard_commit'],d['torch_git_version'])


if __name__=='__main__':unittest.main()
