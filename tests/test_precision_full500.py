from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from rapidtinyrec.precision_full500 import PROTOCOL, PROTOCOL_SHA256, load_protocol, validate_scope, phase_record, validate_phase, validate_repeats
ROOT=Path(__file__).resolve().parents[1]

class Full500ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.protocol=load_protocol(ROOT/PROTOCOL)
        # Synthetic metadata only; no model inference or cohort bootstrap.
        self.users=[dict(cohort_index=i,prompt_tokens=76+5*(i%10)) for i in range(500)]
        self.protocol['cohort']['users_sha256']=hashlib.sha256(json.dumps(self.users,sort_keys=True).encode()).hexdigest()
    def manifest(self,p):
        return dict(experiment='precision_full500_quality',phase=p,variant=self.protocol['phases'][p]['variant'],half=p//2,users=self.users,compiler_artifacts={'reused':p>=2},api_log_staging=True,phase_protocol=phase_record(self.protocol,p,p>=2,self.users),release_verification=dict(schema='precision-release-verification-v1',observed_sha256=self.protocol['provenance']['release_hashes'],elapsed_s=1.,boundary='before_tokenizer_and_model_load'))
    def test_frozen_design_and_four_phase_schedule(self):
        self.assertEqual(hashlib.sha256((ROOT/PROTOCOL).read_bytes()).hexdigest(),PROTOCOL_SHA256)
        for p in range(4):validate_phase(self.manifest(p),self.protocol,self.protocol['provenance']['release_hashes'])
        self.assertEqual(set(self.protocol['phases'][0]['primary_cohort_indices'])|set(self.protocol['phases'][2]['primary_cohort_indices']),set(range(500)))
    def test_protocol_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'protocol';p.write_bytes((ROOT/PROTOCOL).read_bytes()+b' ')
            with self.assertRaises(AssertionError):load_protocol(p)
    def test_scope_is_explicit_full_staged_exclusive(self):
        good=dict(full_cohort=True,full500_quality=True,paired_staged_pilot=False,setup_feasibility=False,api_log_local_root='local')
        validate_scope(**good)
        for k,v in [('full_cohort',False),('full500_quality',False),('paired_staged_pilot',True),('setup_feasibility',True),('api_log_local_root',None)]:
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_scope(**dict(good,**{k:v}))
        validate_scope(full_cohort=False,full500_quality=False,paired_staged_pilot=True,setup_feasibility=False,api_log_local_root='local')
    def test_cohort_order_missing_and_prompt_mutations(self):
        for change in [lambda u:u.reverse(),lambda u:u.pop(),lambda u:u[0].update(prompt_tokens=77),lambda u:u[0].update(cohort_index=1)]:
            u=deepcopy(self.users);change(u)
            with self.assertRaises(AssertionError):phase_record(self.protocol,0,False,u)
    def test_phase_cache_release_scope_mutations(self):
        for change in [lambda m:m.update(experiment='precision_500'),lambda m:m.update(half=0),lambda m:m.update(variant='folded_fp32'),lambda m:m.update(api_log_staging=False),lambda m:m['compiler_artifacts'].update(reused=False),lambda m:m['phase_protocol'].update(graphs=8),lambda m:m['release_verification']['observed_sha256'].pop('Office_ckpt/model.safetensors')]:
            m=deepcopy(self.manifest(2));change(m)
            with self.assertRaises(AssertionError):validate_phase(m,self.protocol,self.protocol['provenance']['release_hashes'])

def run_record():
    return dict(token_ids=[[i,2,3,198,9] for i in range(50)],sequence_logp=[0.]*50,transition_logp=[[0.]*5 for _ in range(50)],beam_indices=[[0,1,2,3,4] for _ in range(50)],saved_score_storage_ptrs=[10,20,30,40,50],graph_logit_storage_ptrs=[60],saved_score_dtypes=['torch.float32']*5)

class UntimedRepeatTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(primary_cohort_indices=[0,2],primary_requests=2,untimed_repeat_requests=2,variant='folded_tf32')
        self.primary=[dict(cohort_index=i,prompt_tokens=76+i,repeat=0,variant='folded_tf32',run=dict(run_record(),elapsed_s=.1)) for i in (0,2)]
        self.repeats=[dict(cohort_index=i,prompt_tokens=76+i,repeat=1,variant='folded_tf32',measurement_role='untimed_stability',run=run_record()) for i in (2,0)]
    def check(self):return validate_repeats(self.primary,self.repeats,self.spec)
    def test_valid_reverse_pass_is_not_quality_or_latency_sample(self):
        r=self.check();self.assertTrue(r['passed']);self.assertEqual(r['comparison']['rank_positions'],100)
        self.assertEqual(r['additional_quality_users'],0);self.assertFalse(r['included_in_primary_latency'])
    def test_missing_duplicate_wrong_order_rejected(self):
        for change in [lambda r:r.pop(),lambda r:r.reverse(),lambda r:r[0].update(cohort_index=0)]:
            self.setUp();change(self.repeats)
            with self.assertRaises(AssertionError):self.check()
    def test_timed_repeat_wrong_role_and_identity_rejected(self):
        for change in [lambda r:r['run'].update(elapsed_s=.01),lambda r:r.update(measurement_role='primary'),lambda r:r.update(repeat=0),lambda r:r.update(variant='folded_fp32'),lambda r:r.update(prompt_tokens=121)]:
            self.setUp();change(self.repeats[0])
            with self.assertRaises(AssertionError):self.check()
    def test_numeric_ancestry_rank_drift_preserved_as_failure(self):
        self.repeats[0]['run']['transition_logp'][0][0]=-0.0
        r=self.check();self.assertFalse(r['passed']);self.assertEqual(r['comparison']['matching_transition_values'],499)
        self.setUp();self.repeats[0]['run']['beam_indices'][0][0]=1
        r=self.check();self.assertFalse(r['passed']);self.assertFalse(r['ancestry_stable'])
        self.setUp();self.repeats[0]['run']['token_ids'][0][0]=999
        self.assertFalse(self.check()['passed'])
    def test_shape_sum_storage_alias_and_nonfinite_rejected(self):
        for change in [lambda r:r['sequence_logp'].__setitem__(0,1.),lambda r:r['transition_logp'][0].pop(),lambda r:r['saved_score_storage_ptrs'].__setitem__(0,60),lambda r:r['saved_score_storage_ptrs'].__setitem__(0,20),lambda r:r['sequence_logp'].__setitem__(0,float('nan'))]:
            self.setUp();change(self.repeats[0]['run'])
            with self.assertRaises(AssertionError):self.check()
if __name__=='__main__':unittest.main()
