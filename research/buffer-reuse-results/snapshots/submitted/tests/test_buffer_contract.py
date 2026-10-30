import copy,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
from rapidtinyrec.buffer_reuse_protocol import phase_schedule,compare_phases,validate_timing,load_protocol,PROTOCOL
from rapidtinyrec.buffer_reuse_measurement import measure_request
from rapidtinyrec.audit_buffer_reuse_outputs import validate_trials,validate_request,compare_user_outputs
from rapidtinyrec.buffer_reuse_outputs import tensor_digest,validate_retained_ownership
from rapidtinyrec.audit_buffer_reuse_dispatch import inspect_dispatch,validate_replay_counts
from fixture_outputs import request
from fixture_dispatch import report,counters
ROOT=Path(__file__).resolve().parents[1]
def trial(spec):
 return dict(schema='released-buffer-reuse-trial-v1',timing=dict(spec,schema='released-buffer-reuse-timing-v1',duration_s=.1 if spec['variant']=='released_tf32' else .09),requests=[request(spec['cohort_index'],0,1,[100+i for i in range(5)])])
class ContractTests(unittest.TestCase):
 def test_timer_includes_generation_sync_excludes_extraction(self):
  events=[];ticks=iter([10.,10.1]);output=object();spec=phase_schedule(1)[0]
  def sync():events.append('sync')
  def clock():events.append('clock');return next(ticks)
  def generate(users):events.append(('generate',users));return output
  def unpack(o,users):self.assertIs(o,output);events.append('unpack');return [dict(cohort_index=users[0])]
  r=measure_request(spec,generate,sync,clock,unpack)
  self.assertEqual(events,['sync','clock',('generate',[1]),'sync','clock','unpack']);self.assertAlmostEqual(r['timing']['duration_s'],.1)
 def test_finite_inventory_strata_and_tail_regression(self):
  load_protocol(ROOT/PROTOCOL)
  phases={p:[trial(s)['timing'] for s in phase_schedule(p)] for p in range(4)}
  self.assertTrue(compare_phases(phases)['timing_screen_passed'])
  self.assertEqual(len(compare_phases(phases)['strata']),9)
  bad=copy.deepcopy(phases);bad[1][-1]['duration_s']=.5
  self.assertFalse(compare_phases(bad)['timing_screen_passed'])
  for value in (float('nan'),float('inf'),0,True):
   with self.assertRaises(ValueError):validate_timing(dict(phases[0][0],duration_s=value))
  bad=copy.deepcopy(phases);bad[2][-1]=bad[2][0]
  with self.assertRaises(ValueError):compare_phases(bad)
 def test_full_score_drift_and_missing_scores_fail(self):
  for p in range(4):self.assertEqual(validate_trials(p,[trial(s) for s in phase_schedule(p)],7)['requests'],12)
  a=trial(phase_schedule(1)[0])['requests'][0];b=copy.deepcopy(a);b['full_score_hashes'][2]['sha256']='b'*64
  self.assertFalse(compare_user_outputs(a,b)['all_full_score_bytes_exact'])
  for change in ('missing','vocab','dtype','hash','alias','ancestry'):
   b=copy.deepcopy(a)
   if change=='missing':b['full_score_hashes'].pop()
   if change=='vocab':b['full_score_hashes'][0]['shape']=[50,6]
   if change=='dtype':b['full_score_hashes'][0]['dtype']='torch.bfloat16'
   if change=='hash':b['full_score_hashes'][0]['sha256']='bad'
   if change=='alias':b['graph_logit_storage_ptrs']=[100]
   if change=='ancestry':b['global_beam_indices'][0][0]=50
   with self.subTest(change=change),self.assertRaises(ValueError):validate_request(b,1,0,1,7)
 def test_byte_hash_distinguishes_signed_zero_and_unselected_values(self):
  a=torch.zeros((50,7));b=a.clone();b[49,6]=-0.
  self.assertNotEqual(tensor_digest(a),tensor_digest(b));self.assertEqual(tensor_digest(a),tensor_digest(a.clone()))
  self.assertEqual(len(tensor_digest(torch.tensor(1))['sha256']),64)
 def test_retained_scores_cannot_alias_across_requests(self):
  from types import SimpleNamespace
  a=SimpleNamespace(scores=[torch.zeros(50,7) for _ in range(5)]);b=SimpleNamespace(scores=[torch.zeros(50,7) for _ in range(5)])
  validate_retained_ownership([([1],a),([2],b)]);b.scores[3]=a.scores[1]
  with self.assertRaises(ValueError):validate_retained_ownership([([1],a),([2],b)])
 def test_both_arms_require_serial_native_dispatch(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for arm in ('released_tf32','reuse_normalized_tf32'):
    self.assertTrue(inspect_dispatch(root,report(root,False),arm)['verified'])
    self.assertFalse(inspect_dispatch(root,report(root,True),arm)['verified'])
    self.assertFalse(inspect_dispatch(root,report(root,False,correlated=False),arm)['verified'])
   for phase in range(4):validate_replay_counts(phase,*counters(False))
 def test_local_benchmark_refuses_before_loading(self):
  env={k:v for k,v in os.environ.items() if k not in ('SLURM_JOB_ID','OSCAR_RUN_DIR')}
  cmd=[sys.executable,'-m','rapidtinyrec.benchmark_buffer_reuse','--phase','0']
  for k in ('release-root','reference-results','output-root','api-log-local-root'):cmd+=['--'+k,'/missing']
  z=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
  self.assertNotEqual(z.returncode,0);self.assertIn('actual allocated job',z.stderr)
if __name__=='__main__':unittest.main()
