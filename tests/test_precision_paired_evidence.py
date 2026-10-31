"""Verify complete paired-pilot archive and keep pilot eligibility distinct from adoption."""
import hashlib
import json
from pathlib import Path
import tarfile
import unittest

ROOT=Path(__file__).resolve().parents[1]/'benchmarks/precision-paired-staged-pilot'


class PairedEvidenceTests(unittest.TestCase):
    def test_every_raw_and_submitted_member_is_preserved(self):
        manifest=json.loads((ROOT/'archive-manifest.json').read_text());members={}
        for archive in manifest['archives']:
            path=ROOT/archive['path'];self.assertLess(path.stat().st_size,48*1024*1024)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),archive['sha256'])
            with tarfile.open(path,'r:gz') as tf:
                for member in tf.getmembers():
                    self.assertTrue(member.isfile());self.assertNotIn(member.name,members)
                    data=tf.extractfile(member).read();expected=manifest['files'][member.name]
                    self.assertEqual(len(data),expected['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),expected['sha256'])
                    members[member.name]=data
        self.assertEqual(len(members),538);self.assertEqual(set(members),set(manifest['files']))
        raw=json.loads(members['review/precision-paired-staged-raw-manifest.json'])['files']
        self.assertEqual(len(raw),447)
        self.assertEqual({n[4:] for n in members if n.startswith('raw/')},{r['path'] for r in raw})
        for row in raw:self.assertEqual(hashlib.sha256(members['raw/'+row['path']]).hexdigest(),row['sha256'])
        record=json.loads(members['run-record.json']);self.assertEqual(len(record['sha256']),75)
        for name,sha in record['sha256'].items():self.assertEqual(hashlib.sha256(members['source-snapshot/'+name]).hexdigest(),sha)
        self.assertEqual((ROOT/'audit.json').read_bytes(),members['audit.json'])

    def test_passed_pilot_preserves_drift_and_cannot_establish_noninferiority(self):
        a=json.loads((ROOT/'audit.json').read_text());s=json.loads((ROOT/'summary.json').read_text())
        self.assertEqual(a['schema'],'precision-paired-staged-audit-v1')
        for key in ('integrity_audit_passed','control_exact','stability_exact','dispatch_verified','validity_passed','eligible_for_full_quality_review'):self.assertIs(a[key],True)
        self.assertTrue(a['performance_gate']['passed']);self.assertFalse(a['candidate_bit_exact'])
        self.assertEqual(a['unique_users'],4);self.assertEqual(a['paired_requests'],24)
        self.assertEqual(a['comparison']['matching_rank_positions'],1182)
        self.assertEqual(a['comparison']['matching_sequence_values'],0)
        self.assertEqual(a['comparison']['matching_transition_values'],0)
        self.assertEqual(len(a['quality']['per_user']),4)
        for result in (a,s,a['quality']):
            self.assertIs(result['promotion'],False);self.assertIs(result['noninferiority_established'],False)
        self.assertEqual([p['phase'] for p in a['phases']],[0,1,2,3])
        wrapper=a['provenance']['wrapper'];self.assertEqual(len(wrapper['recovery']['locations']),4)
        self.assertTrue(wrapper['child']['completed_normally']);self.assertLess(wrapper['child']['elapsed_s'],1620)
