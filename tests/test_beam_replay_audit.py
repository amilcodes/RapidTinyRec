import copy
import hashlib
import math
import struct
import unittest

from rapidtinyrec.audit_beam_helpers import (
    METHODS, FINISHED, HASHES, input_spec, output_spec, storage_bytes,
    audit_report, audit_beam_phase, raw_comparison, exact_comparison,
)
from rapidtinyrec.audit_beam_replay import quality, VARIANTS


def report():
    events=[]
    for n in (76,121):
        for step in range(5):
            for method in METHODS:
                spec=input_spec(method,n)
                signature=[[key,'tensor',shape,[math.prod(shape[i+1:]) for i in range(len(shape))],dtype,'cuda:0'] for key,(shape,dtype) in spec.items()]
                signature.append(['num_beams','int',50])
                if method==FINISHED:
                    signature.extend([['cur_len','int',n+step],['decoder_prompt_len','int',n],['length_penalty','float',0.],['early_stopping','bool',False]])
                events.append(dict(method=method,prompt_tokens=n,step=step,signature=signature,
                    input_bytes=storage_bytes(spec.values()),output_bytes=storage_bytes(output_spec(method,n)),capture_and_warmup_s=1.))
    a=sum(e['input_bytes'] for e in events);b=sum(e['output_bytes'] for e in events)
    return dict(graphs=20,frozen=True,disjoint_step_storage=True,active_request=False,aborted_requests=0,
        helper_source_sha256=HASHES,events=events,input_logical_bytes=a,output_logical_bytes=b,
        unique_input_output_storage_bytes=a+b,capture_and_warmup_s=20.,requests=8,replays=80,staged_bytes=4*a)


def comparison(shape,dtype):
    return dict(shape=shape,dtype=dtype,values=math.prod(shape),matching_bits=math.prod(shape),max_abs_finite=0.,
        finite_mask_equal=True,no_nan=True,left_sha256='a'*64,right_sha256='a'*64)


def fixture():
    before=report();after=copy.deepcopy(before)
    rows=[dict(prompt_tokens=n) for _ in range(3) for n in (76,76,121,121)]
    after.update(requests=20,replays=200,staged_bytes=before['staged_bytes']+6*before['input_logical_bytes'])
    setup=dict(beam_replay=before,score_ownership_checks=8,warmup_cohort_indices=[1,2,3,4],combined_setup_and_warmup_s=30.)
    summary=dict(beam_replay=after,score_ownership_checks=20)
    ownership=[]
    for _ in range(10):
        for n in (76,121):
            ownership.append(dict(prompt_tokens=n,steps=5,helper_calls=10,distinct_helper_output_storages=35,
                distinct_saved_score_storages=5,saved_score_shapes=[[50,152265]]*5,
                saved_outputs_independent_of_graphs=True,lease_open_during_extraction=True,
                final_output_comparisons=[comparison([50,n+5],'torch.int64'),comparison([50],'torch.float32'),comparison([50,5],'torch.int32')]))
    return setup,summary,ownership,rows


class BeamAuditTests(unittest.TestCase):
    def test_synthetic_initial_slice_stride_is_source_specific(self):
        probe=report()
        for event in probe['events']:
            if event['method']==FINISHED and event['step']==0:
                for item in event['signature']:
                    if item[0] in ('sequences','beam_indices'):
                        item[3][0] *= 2
        audit_report(probe,[76,121],synthetic_probe=True)
        with self.assertRaises(AssertionError):audit_report(probe,[76,121])
        with self.assertRaises(AssertionError):audit_report(report(),[76,121],synthetic_probe=True)
        for name,axis in [('sequences',0),('sequences',1),('beam_indices',0),('beam_indices',2)]:
            changed=copy.deepcopy(probe)
            item=next(x for x in changed['events'][1]['signature'] if x[0]==name)
            item[3][axis] += 1
            with self.assertRaises(AssertionError):audit_report(changed,[76,121],synthetic_probe=True)

    def test_pinned_shapes_scalar_steps_storage_and_captures(self):
        audit_report(report(),[76,121])
        for mutate in (
            lambda r:r['events'][1]['signature'][-4].__setitem__(2,99),
            lambda r:r['events'][0].update(input_bytes=1),
            lambda r:r.update(unique_input_output_storage_bytes=1),
            lambda r:r.update(aborted_requests=1),
            lambda r:r.update(helper_source_sha256={}),
            lambda r:r['events'].pop(),
        ):
            r=copy.deepcopy(report());mutate(r)
            with self.assertRaises(AssertionError):audit_report(r,[76,121])

    def test_timed_capture_staging_and_lease_cannot_be_omitted(self):
        self.assertTrue(audit_beam_phase(*fixture(),True)['final_outputs_exact'])
        for mutate in (
            lambda a,z,o:z['beam_replay']['events'][0].update(capture_and_warmup_s=2.),
            lambda a,z,o:z['beam_replay'].update(staged_bytes=z['beam_replay']['staged_bytes']-1),
            lambda a,z,o:o[-1].update(distinct_helper_output_storages=34),
            lambda a,z,o:o[-1].update(lease_open_during_extraction=False),
            lambda a,z,o:o[-1].update(saved_outputs_independent_of_graphs=False),
        ):
            a,z,o,rows=fixture();mutate(a,z,o)
            with self.assertRaises(AssertionError):audit_beam_phase(a,z,o,rows,True)

    def test_changed_ancestry_or_score_is_a_negative_result(self):
        for index in (1,2):
            a,z,o,rows=fixture();c=o[-1]['final_output_comparisons'][index]
            c.update(matching_bits=c['values']-1,right_sha256='b'*64,max_abs_finite=1.)
            result=audit_beam_phase(a,z,o,rows,True)
            self.assertFalse(result['final_outputs_exact'])
            self.assertEqual(result['final_output_comparison']['matching_bits'],result['final_output_comparison']['values']-1)

    def test_raw_signed_zero_and_ancestor_bits_recomputed(self):
        for dtype,fmt,a,b in [('torch.float32','f',[0.,float('-inf')],[-0.,float('-inf')]),('torch.int32','i',[1,2],[2,1])]:
            left=b''.join(struct.pack('<'+fmt,x) for x in a);right=b''.join(struct.pack('<'+fmt,x) for x in b)
            c=comparison([2],dtype)
            c.update(matching_bits=1 if fmt=='f' else 0,max_abs_finite=0. if fmt=='f' else 1.,
                left_sha256=hashlib.sha256(left).hexdigest(),right_sha256=hashlib.sha256(right).hexdigest())
            raw_comparison(c,a,b);self.assertFalse(exact_comparison(c))
            c['matching_bits']=2
            with self.assertRaises(AssertionError):raw_comparison(c,a,b)

    def test_quality_uses_unique_users(self):
        users=[dict(cohort_index=7,target_sid='<a><b><c>')]
        row=dict(cohort_index=7,runs={v:dict(token_ids=[[1,2,3]]+[[0,0,0]]*49) for v in VARIANTS})
        result=quality([row]*6,users,{'<a>':1,'<b>':2,'<c>':3})
        self.assertEqual(result['independent_users'],1)
        self.assertEqual(result['paired_mean_changes']['50']['ndcg'],0.)


if __name__=='__main__':unittest.main()
