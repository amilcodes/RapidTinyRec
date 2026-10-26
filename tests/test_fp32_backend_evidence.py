"""Check the completed negative result and complete immutable evidence archive."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/diagnostics/fp32-backend'
sha=lambda b:hashlib.sha256(b).hexdigest()

class BackendEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((ROOT/'summary.json').read_text())
        cls.manifest=json.loads((ROOT/'archive-manifest.json').read_text())
        cls.source=json.loads((ROOT/'source.json').read_text())

    def test_every_archived_member_and_frozen_file_matches_hash(self):
        seen=set();raw=0;snapshot={}
        for shard in self.manifest['archives']:
            p=ROOT/shard['path'];self.assertLess(p.stat().st_size,48*1024*1024)
            self.assertEqual(sha(p.read_bytes()),shard['sha256'])
            with tarfile.open(p,'r:gz') as archive:
                members=archive.getmembers();self.assertEqual(len(members),shard['files'])
                for member in members:
                    self.assertTrue(member.isfile());self.assertNotIn(member.name,seen);seen.add(member.name)
                    entry=self.manifest['files'][member.name];data=archive.extractfile(member).read()
                    self.assertEqual((len(data),sha(data)),(entry['bytes'],entry['sha256']))
                    self.assertEqual(entry['archive'],shard['path'])
                    raw+=member.name.startswith('raw/')
                    if member.name.startswith('source-snapshot/'):
                        snapshot[member.name.removeprefix('source-snapshot/')]=sha(data)
        self.assertEqual(seen,set(self.manifest['files']));self.assertEqual(len(seen),320);self.assertEqual(raw,246)
        for name,digest in self.source['submitted_source_sha256'].items():self.assertEqual(snapshot[name],digest)
        self.assertEqual(len(self.source['submitted_source_sha256']),47)
        self.assertIn('review/review_fp32_backend.py',seen)
        self.assertIn('prior-startup-failure/published/evidence-001.tar.gz',seen)

    def test_negative_gate_and_exact_controls_are_not_hidden(self):
        s=self.summary
        for key in ['integrity_audit_passed','control_exact','stability_exact','dispatch_verified']:self.assertIs(s[key],True)
        for key in ['candidate_exact','diagnostic_gate','end_to_end_promotion','ranking_quality_measured','generation_latency_measured']:self.assertIs(s[key],False)
        self.assertEqual(len(s['comparisons']),1728)
        for role in ['gate','up','down']:
            rows=[r for r in s['comparisons'] if r['category']=='versus_default' and r['backend']=='cublaslt' and r['state']=='original' and r['role']==role]
            self.assertEqual(len(rows),12)
            if role=='down':
                self.assertTrue(all(r['matching_bits']<r['values'] for r in rows))
                self.assertEqual(sum(r['matching_bits'] for r in rows),63210)
                self.assertEqual(sum(r['values'] for r in rows),921600)
                self.assertEqual(max(r['max_abs_finite'] for r in rows),3.814697265625e-05)
            else:self.assertTrue(all(r['matching_bits']==r['values'] for r in rows))

    def test_timing_and_dispatch_preserve_all_regressions(self):
        s=self.summary;table={(r['cohort_index'],r['layer'],r['role'],r['backend']):r for r in s['timing_cases']}
        self.assertEqual(len(table),108);self.assertEqual(len(s['dispatch']),36)
        self.assertTrue(all(r['verified'] for r in s['dispatch']))
        for u in [1,2,6,10]:
            for l in [0,13,27]:
                for metric in ['median_us','p95_us']:
                    self.assertGreater(table[u,l,'down','cublaslt'][metric],table[u,l,'down','default'][metric])
        self.assertEqual((s['candidate_faster_median_cases'],s['candidate_faster_p95_cases']),(24,13))
        for name,count in [('comparisons.csv',1728),('timing.csv',108),('dispatch.csv',36)]:
            with (ROOT/name).open() as f:self.assertEqual(len(list(csv.DictReader(f))),count)

    def test_published_files_and_allocation_match_provenance(self):
        for name,digest in self.source['evidence_sha256'].items():self.assertEqual(sha((ROOT/name).read_bytes()),digest)
        self.assertEqual(self.source['measured_repository_commit'],'aa9ba129ed0dc4020bbd8184a9918dd1c9dd01c8')
        self.assertEqual(self.source['job_id'],'7956300')
        self.assertEqual(self.source['resources']['cpus'],4)
        self.assertIn('7956300|COMPLETED|4|4|32G|billing=112,cpu=4,gres/gpu=1,mem=32G,node=1|00:00:45',(ROOT/'resource-accounting.txt').read_text())

if __name__=='__main__':unittest.main()
