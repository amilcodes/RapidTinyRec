import copy,importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from rapidtinyrec.buffer_reuse_outputs import unpack_requests
from rapidtinyrec.audit_buffer_reuse_outputs import validate_request,compare_user_outputs
from rapidtinyrec.audit_buffer_reuse import finalize_gates
class NewGateTests(unittest.TestCase):
 def test_full_output_producer_and_independent_checker(self):
  output=SimpleNamespace(sequences=torch.ones(50,11,dtype=torch.int64),sequences_scores=torch.full((50,),-15.),beam_indices=torch.arange(5).expand(50,5).clone(),scores=[torch.full((50,7),-float(i)) for i in range(1,6)])
  model=SimpleNamespace(compute_transition_scores=lambda *a,**k:torch.tensor([-1.,-2.,-3.,-4.,-5.]).expand(50,5))
  a=unpack_requests(model,output,[1],6)[0];validate_request(a,1,0,1,7)
  output.scores[2][49,6]=-3.000001
  b=unpack_requests(model,output,[1],6)[0];validate_request(b,1,0,1,7)
  diff=compare_user_outputs(a,b)
  self.assertTrue(diff['scores_bit_exact']);self.assertFalse(diff['all_full_score_bytes_exact'])
 def test_performance_cannot_override_full_score_failure_or_missing_pair(self):
  x=dict(integrity_audit_passed=True,control_exact=True,stability_exact=True,dispatch_verified=True,validity_passed=True,timing=dict(timing_screen_passed=True),cross_variant_comparisons=[dict(tokens_exact=True,scores_bit_exact=True,local_ancestry_exact=True,all_full_score_bytes_exact=True) for _ in range(24)])
  self.assertTrue(finalize_gates(copy.deepcopy(x))['eligible_for_full500'])
  for change in ('drift','missing_pair','missing_key','nonboolean'):
   y=copy.deepcopy(x)
   if change=='drift':y['cross_variant_comparisons'][0]['all_full_score_bytes_exact']=False
   if change=='missing_pair':y['cross_variant_comparisons'].pop()
   if change=='missing_key':y['cross_variant_comparisons'][0].pop('all_full_score_bytes_exact')
   if change=='nonboolean':y['cross_variant_comparisons'][0]['all_full_score_bytes_exact']=1
   self.assertFalse(finalize_gates(y)['execution_audit_passed'])
 def test_guard_queue_budget_intent_and_active_run_fail_closed(self):
  p=Path(__file__).resolve().parents[2]/'submit_buffer_reuse.py';s=importlib.util.spec_from_file_location('buffer_guard',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  state=dict(status='ACTIVE',active_remote=None,remote_allocations_submitted=0,resources=dict(max_allocations_per_24h=6))
  m.validate_window(state,'',False,[])
  for queue,intent,records,delta in [(None,False,[],{}),('123 RUNNING other',False,[],{}),('',True,[],{}),('',False,['run'],{}),('',False,[],dict(active_remote={'job':'123'})),('',False,[],dict(remote_allocations_submitted=6))]:
   with self.assertRaises(ValueError):m.validate_window(dict(state,**delta),queue,intent,records)
if __name__=='__main__':unittest.main()
