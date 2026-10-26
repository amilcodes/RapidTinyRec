"""Reject lost outputs, timed capture/staging omissions and tolerance-style passes."""
import copy
import unittest

from rapidtinyrec.audit_constraint_replay import audit_probe, audit_constraint_phase, quality, VARIANTS


def probe_fixture():
    rows=[]
    for scenario in ('valid','unknown','mixed','valid_repeat'):
        for step in range(5):
            rows.append(dict(scenario=scenario,step=step,values=50*152265,matching_bits=50*152265,
                max_abs_finite=0.,finite_mask_equal=True,no_nan=True,score_shape=[50,152265],
                prompt_tokens=76 if scenario in ('valid','mixed') else 121,
                unknown_fallback_checked_rows={'unknown':50,'mixed':25}.get(scenario,0),unknown_fallback_correct=True))
    return dict(complete=True,independent_step_outputs=True,scenarios=['valid','unknown','mixed','valid_repeat'],
        graph_direct=rows,retained_after_five_steps=copy.deepcopy(rows),
        replay=dict(requests=4,replays=20,frozen=True,active_request=False))


def phase_fixture():
    score=50*152265*4;suffix=50*3*8
    before=dict(graphs=5,frozen=True,disjoint_step_storage=True,active_request=False,
        input_logical_bytes=5*(score+suffix),output_logical_bytes=5*score,
        unique_input_output_storage_bytes=5*(2*score+suffix),
        events=[dict(step=i,score_shape=[50,152265],capture_and_warmup_s=1.) for i in range(5)],
        capture_and_warmup_s=5.,replays=60,requests=12,staged_bytes=60*(score+suffix))
    after=copy.deepcopy(before);after.update(replays=120,requests=24,staged_bytes=120*(score+suffix))
    setup=dict(constraint_replay=before,score_ownership_checks=8,warmup_cohort_indices=[1,2,6,10],combined_setup_and_warmup_s=10.)
    summary=dict(constraint_replay=after,score_ownership_checks=20)
    ownership=[dict(steps=5,distinct_saved_score_storages=5,saved_outputs_match_graphs=True,lease_open_during_extraction=True) for _ in range(20)]
    return setup,summary,ownership


class ConstraintAuditTests(unittest.TestCase):
    def test_one_bit_failure_is_preserved_as_negative_exactness(self):
        good=probe_fixture();self.assertTrue(audit_probe(good)['exact'])
        for field in ('graph_direct','retained_after_five_steps'):
            changed=copy.deepcopy(good);changed[field][0]['matching_bits']-=1
            changed[field][0]['max_abs_finite']=2**-23
            report=audit_probe(changed)
            self.assertFalse(report['exact'])
            self.assertEqual(report[field]['matching_bits'],report[field]['values']-1)
        changed=copy.deepcopy(good);changed['graph_direct'][5]['unknown_fallback_correct']=False
        self.assertFalse(audit_probe(changed)['exact'])

    def test_missing_scenario_or_step_cannot_pass(self):
        changed=probe_fixture();changed['graph_direct'].pop()
        with self.assertRaises(AssertionError):audit_probe(changed)

    def test_capture_staging_and_output_lifetime_gates(self):
        s,z,o=phase_fixture();self.assertIsNotNone(audit_constraint_phase(s,z,o,12,True))
        for mutate in (lambda s,z,o:z['constraint_replay']['events'][0].update(capture_and_warmup_s=2.),
                       lambda s,z,o:z['constraint_replay'].update(staged_bytes=z['constraint_replay']['staged_bytes']-1),
                       lambda s,z,o:o[-1].update(distinct_saved_score_storages=1),
                       lambda s,z,o:o[-1].update(lease_open_during_extraction=False),
                       lambda s,z,o:z['constraint_replay'].update(active_request=True)):
            s,z,o=phase_fixture();mutate(s,z,o)
            with self.assertRaises(AssertionError):audit_constraint_phase(s,z,o,12,True)

    def test_quality_counts_unique_users_and_reports_paired_changes(self):
        users=[dict(cohort_index=1,target_sid='<a><b><c>'),dict(cohort_index=2,target_sid='<a><b><c>')]
        def run(rank):
            seqs=[[0,0,i] for i in range(50)]
            if rank is not None:seqs[rank-1]=[1,2,3]
            return dict(token_ids=seqs)
        first=dict(cohort_index=1,runs={VARIANTS[0]:run(1),VARIANTS[1]:run(2)})
        second=dict(cohort_index=2,runs={VARIANTS[0]:run(None),VARIANTS[1]:run(1)})
        result=quality([first,first,second,second],users,{'<a>':1,'<b>':2,'<c>':3})
        self.assertEqual(result['independent_users'],2)
        self.assertEqual(result['paired_mean_changes']['1']['hit_rate'],0.)
        self.assertEqual(result['paired_mean_changes']['50']['hit_rate'],0.5)
        self.assertEqual(result['paired_mean_changes']['50']['mrr'],0.25)


if __name__=='__main__':unittest.main()
