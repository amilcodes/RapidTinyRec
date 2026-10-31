import hashlib,json,unittest
from copy import deepcopy
from pathlib import Path
from rapidtinyrec.bf16_full500 import PROTOCOL,load_protocol,validate_protocol,phase_record
ROOT=Path(__file__).resolve().parents[1]
class Full500ProtocolTests(unittest.TestCase):
    def setUp(self):self.p=load_protocol(ROOT/PROTOCOL)
    def test_frozen_schedule_covers_each_user_once_per_policy(self):
        for variant in ['folded_tf32','folded_bf16']:
            self.assertEqual(sorted(u for s in self.p['phases'] if s['variant']==variant for u in s['primary_cohort_indices']),list(range(500)))
    def test_rejects_phase_relabel_and_missing_repeat(self):
        for field,value in [('variant','folded_fp32'),('untimed_repeat_requests',0),('graphs',8),('fresh_process',False),('disk_cache','cold')]:
            p=deepcopy(self.p);p['phases'][2][field]=value
            with self.assertRaises(AssertionError):validate_protocol(p)
    def test_rejects_pilot_cohort_duplicate_user_or_shortened_schedule(self):
        for mutate in [lambda p:p['cohort'].update(indices=[1,2,6,10]),lambda p:p['phases'][0]['primary_cohort_indices'].__setitem__(0,2),lambda p:p['phases'].pop()]:
            p=deepcopy(self.p);mutate(p)
            with self.assertRaises(AssertionError):validate_protocol(p)
    def test_rejects_extended_deadline_local_audit_or_promotion(self):
        for mutate in [lambda p:p['budget'].update(global_work_deadline_s=1800),lambda p:p['quality'].update(bootstrap_execution='local'),lambda p:p.update(promotion=True)]:
            p=deepcopy(self.p);mutate(p)
            with self.assertRaises(AssertionError):validate_protocol(p)
    def test_phase_identity_validates_all_user_metadata(self):
        p=deepcopy(self.p);lengths=p['cohort']['lengths'];users=[dict(cohort_index=i,prompt_tokens=lengths[i%len(lengths)]) for i in range(500)]
        p['cohort']['users_sha256']=hashlib.sha256(json.dumps(users,sort_keys=True).encode()).hexdigest()
        record=phase_record(p,2,True,users);self.assertEqual(record['schema'],'bf16-full500-phase-v1')
        self.assertEqual(record['phase']['variant'],'folded_bf16')
        users[4]['prompt_tokens']+=1
        with self.assertRaises(AssertionError):phase_record(p,2,True,users)
    def test_protocol_has_concrete_catalog_and_checkpoint_hashes(self):
        pins=json.loads((ROOT/'rapidtinyrec/quality_resources.json').read_text())
        self.assertEqual(self.p['provenance']['resource_hashes'],pins)
        p=deepcopy(self.p);p['provenance']['resource_hashes']='Use existing hashes'
        with self.assertRaises(AssertionError):validate_protocol(p)
    def test_rejects_wrong_reuse_or_phase_boolean(self):
        with self.assertRaises(AssertionError):phase_record(self.p,True,False,[])
        with self.assertRaises(AssertionError):phase_record(self.p,2,False,[])
if __name__=='__main__':unittest.main()
