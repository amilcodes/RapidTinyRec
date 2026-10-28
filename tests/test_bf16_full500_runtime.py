from copy import deepcopy
import hashlib,json,unittest,weakref
from pathlib import Path
from rapidtinyrec.bf16_full500 import PROTOCOL,load_protocol,phase_record,select_users,primary_trials,collect_repeats,validate_phase,validate_repeats
from test_precision_full500 import run_record
ROOT=Path(__file__).resolve().parents[1]
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.p=load_protocol(ROOT/PROTOCOL)
        self.users=[dict(cohort_index=i,prompt_tokens=self.p['cohort']['lengths'][i%10]) for i in range(500)]
        self.p['cohort']['users_sha256']=hashlib.sha256(json.dumps(self.users,sort_keys=True).encode()).hexdigest()
    def test_full_cohort_and_warmup_keep_all_ten_lengths(self):
        users,warm=select_users(self.users,self.p)
        self.assertEqual(users,self.users);self.assertEqual([u['prompt_tokens'] for u in warm],self.p['cohort']['lengths'])
        for phase in range(4):self.assertEqual([u['cohort_index'] for _,u in primary_trials(users,self.p,phase)],list(range(phase//2,500,2)))
    def test_repeat_reverse_sync_release_save_no_timing(self):
        refs=[];saved=[];events=[]
        class Output:pass
        def generate(u):
            self.assertTrue(all(r() is None for r in refs));o=Output();refs.append(weakref.ref(o));events.append(('generate',u['cohort_index']));return o
        def unpack(o,u):events.append(('unpack',u['cohort_index']));return {'token_ids':[[u['cohort_index']]]}
        def sync():events.append(('sync',))
        def save(row):self.assertTrue(all(r() is None for r in refs));saved.append(row)
        rows=collect_repeats(self.users,self.p,2,generate,unpack,sync,save)
        self.assertEqual(len(rows),250);self.assertEqual(rows,saved)
        self.assertEqual([r['cohort_index'] for r in rows],list(range(499,0,-2)))
        self.assertTrue(all(r['variant']=='folded_bf16' and r['measurement_role']=='untimed_stability' and 'elapsed_s' not in r['run'] for r in rows))
        self.assertEqual(events[:3],[('generate',499),('sync',),('unpack',499)])
    def test_repeat_failure_preserves_written_prefix_never_retries(self):
        seen=[];saved=[]
        def generate(u):
            seen.append(u['cohort_index'])
            if len(seen)==3:raise RuntimeError('injected generation failure')
            return object()
        with self.assertRaises(RuntimeError):collect_repeats(self.users,self.p,0,generate,lambda o,u:{},lambda:None,saved.append)
        self.assertEqual(seen,[498,496,494]);self.assertEqual([r['cohort_index'] for r in saved],[498,496])
    def test_manifest_requires_distinct_identity_and_catalog_pins(self):
        verification=lambda hashes:dict(schema='precision-release-verification-v1',observed_sha256=hashes,elapsed_s=1.,boundary='before_tokenizer_and_model_load')
        m=dict(experiment='bf16_vs_tf32_full500',phase=1,half=0,variant='folded_bf16',api_log_staging=True,users=self.users,compiler_artifacts={'reused':False},phase_protocol=phase_record(self.p,1,False,self.users),release_verification=verification(self.p['provenance']['release_hashes']),quality_resource_verification=verification(self.p['provenance']['resource_hashes']))
        validate_phase(m,self.p)
        for change in [lambda m:m.update(experiment='bf16_vs_tf32_pilot'),lambda m:m.update(variant='folded_tf32'),lambda m:m['quality_resource_verification'].update(observed_sha256={})]:
            bad=deepcopy(m);change(bad)
            with self.assertRaises(AssertionError):validate_phase(bad,self.p)
    def test_primary_timing_role_and_repeat_drift(self):
        spec=dict(primary_cohort_indices=[0],primary_requests=1,untimed_repeat_requests=1,variant='folded_bf16')
        primary=[dict(cohort_index=0,prompt_tokens=76,repeat=0,variant='folded_bf16',measurement_role='primary_timed',run=dict(run_record(),elapsed_s=.1))]
        repeat=[dict(cohort_index=0,prompt_tokens=76,repeat=1,variant='folded_bf16',measurement_role='untimed_stability',run=run_record())]
        self.assertTrue(validate_repeats(primary,repeat,spec)['passed'])
        repeat[0]['run']['transition_logp'][0][0]=-0.0
        self.assertFalse(validate_repeats(primary,repeat,spec)['passed'])
        primary[0]['measurement_role']='untimed_stability'
        with self.assertRaises(AssertionError):validate_repeats(primary,repeat,spec)
if __name__=='__main__':unittest.main()
