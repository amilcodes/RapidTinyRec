import copy
import unittest
from rapidtinyrec.audit_beam_wave_outputs import validate_trial,validate_trials,compare_user_outputs
from rapidtinyrec.beam_wave_protocol import phase_schedule


def request(user,slot,batch,pointers):
    span=50*7*4
    return dict(cohort_index=user,request_slot=slot,original_batch_requests=batch,original_beam_row_offset=50*slot,
        token_ids=[[1,2,3,4,5] for _ in range(50)],sequence_logp=[-15.]*50,
        transition_logp=[[-1.,-2.,-3.,-4.,-5.] for _ in range(50)],
        beam_indices=[[0,1,2,3,4] for _ in range(50)],global_beam_indices=[[x+50*slot for x in range(5)] for _ in range(50)],
        saved_score_storage_ptrs=pointers,graph_logit_storage_ptrs=[999],saved_score_dtypes=['torch.float32']*5,
        score_row_extents=[dict(storage_ptr=p,byte_start=slot*span,byte_end=(slot+1)*span,storage_bytes=batch*span) for p in pointers])


def trial(spec):
    wave=spec['variant']=='beam_wave_tf32'
    return dict(schema='two-request-beam-wave-trial-v1',timing=dict(spec,schema='two-request-beam-wave-timing-v1',
        arrival_offsets_s=[0,0],completion_offsets_s=[.12,.12] if wave else [.1,.2],pair_duration_s=.12 if wave else .2),
        requests=[request(user,slot if wave else 0,2 if wave else 1,[100+i+(0 if wave else slot*10) for i in range(5)])
                  for slot,user in enumerate(spec['cohort_indices'])])


class WaveOutputAuditTests(unittest.TestCase):
    def test_complete_serial_and_wave_inventory(self):
        for phase in range(4):
            result=validate_trials(phase,[trial(s) for s in phase_schedule(phase)],7)
            self.assertEqual(result['requests'],12);self.assertIsNone(result['full_audit_passed'])
            self.assertFalse(result['sid_validity_checked'])

    def test_cross_request_global_ancestry_not_hidden_by_local_indices(self):
        t=trial(phase_schedule(1)[0]);t['requests'][1]['global_beam_indices'][0][0]=0
        with self.assertRaisesRegex(ValueError,'Global ancestry'):validate_trial(t,7)

    def test_original_five_step_sum_and_finite_scores(self):
        for value in [-14.,float('nan'),float('inf')]:
            t=trial(phase_schedule(0)[0]);t['requests'][0]['sequence_logp'][0]=value
            with self.assertRaises(ValueError):validate_trial(t,7)

    def test_wave_shared_storage_requires_disjoint_full_vocabulary_rows(self):
        t=trial(phase_schedule(1)[0]);t['requests'][1]['score_row_extents'][0]['byte_start']=0
        with self.assertRaises(ValueError):validate_trial(t,7)
        with self.assertRaises(ValueError):validate_trial(trial(phase_schedule(1)[0]),8)

    def test_serial_retained_outputs_must_not_alias(self):
        t=trial(phase_schedule(0)[0]);t['requests'][1]['saved_score_storage_ptrs']=t['requests'][0]['saved_score_storage_ptrs'][:]
        for a,b in zip(t['requests'][0]['score_row_extents'],t['requests'][1]['score_row_extents']):b['storage_ptr']=a['storage_ptr']
        with self.assertRaisesRegex(ValueError,'serial request'):validate_trial(t,7)

    def test_graph_alias_and_missing_rank_fail(self):
        t=trial(phase_schedule(1)[0]);t['requests'][0]['graph_logit_storage_ptrs']=[100]
        with self.assertRaisesRegex(ValueError,'graph logits'):validate_trial(t,7)
        t=trial(phase_schedule(1)[0]);t['requests'][0]['token_ids'].pop()
        with self.assertRaisesRegex(ValueError,'50ranks'):validate_trial(t,7)

    def test_pair_order_and_duplicate_trial_rejected(self):
        t=trial(phase_schedule(1)[0]);t['requests'].reverse()
        with self.assertRaisesRegex(ValueError,'identity'):validate_trial(t,7)
        rows=[trial(s) for s in phase_schedule(1)];rows[-1]=rows[0]
        with self.assertRaises(ValueError):validate_trials(1,rows,7)

    def test_reversal_compares_local_ancestry_after_validation(self):
        a=trial(phase_schedule(1)[0]);b=trial(phase_schedule(1)[2]);validate_trial(a,7);validate_trial(b,7)
        self.assertNotEqual(a['requests'][0]['global_beam_indices'],b['requests'][1]['global_beam_indices'])
        self.assertTrue(all(compare_user_outputs(a['requests'][0],b['requests'][1]).values()))
        b['requests'][1]['transition_logp'][0][0]=-1.0000001192092896
        self.assertFalse(compare_user_outputs(a['requests'][0],b['requests'][1])['scores_bit_exact'])


if __name__=='__main__':unittest.main()
