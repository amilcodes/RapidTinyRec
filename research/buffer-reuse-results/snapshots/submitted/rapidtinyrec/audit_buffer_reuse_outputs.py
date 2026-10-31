"""New exact engineering contract including all full-vocabulary score bytes."""
import re
from .audit_beam_wave_outputs import validate_request as original_validate,compare_user_outputs as original_compare
from .buffer_reuse_protocol import validate_timing,validate_phase_rows

def validate_request(row,user,slot,batch_requests,vocab_size):
 if slot!=0 or batch_requests!=1:raise ValueError('Single native request only')
 original_validate(row,user,slot,batch_requests,vocab_size)
 hashes=row['full_score_hashes']
 if len(hashes)!=5:raise ValueError('Five full score digests required')
 for r in hashes:
  if r['shape']!=[50,vocab_size] or r['dtype']!='torch.float32' or not re.fullmatch('[0-9a-f]{64}',r['sha256']):raise ValueError('Full score digest scope differs')

def validate_request_group(requests,users,wave,vocab_size):
 if wave or len(requests)!=len(users) or len(set(users))!=len(users):raise ValueError('Invalid serial inventory')
 seen=set()
 for row,u in zip(requests,users):
  validate_request(row,u,0,1,vocab_size)
  ptrs=set(row['saved_score_storage_ptrs'])
  if seen&ptrs:raise ValueError('Simultaneously retained score outputs alias')
  seen.update(ptrs)

def validate_trials(phase,trials,vocab_size):
 timings=[]
 for t in trials:
  if t['schema']!='released-buffer-reuse-trial-v1':raise ValueError('Wrong trial identity')
  r=validate_timing(t['timing']);validate_request_group(t['requests'],[r['cohort_index']],False,vocab_size);timings.append(r)
 validate_phase_rows(phase,timings)
 return dict(phase=phase,requests=len(trials),five_full_score_hashes=True,original_score_shape_and_sum=True,request_ancestry_mapping=True)

def compare_user_outputs(left,right):
 result=original_compare(left,right)
 result['all_full_score_bytes_exact']=left['full_score_hashes']==right['full_score_hashes']
 return result
