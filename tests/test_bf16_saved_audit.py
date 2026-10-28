import os
import unittest
from unittest.mock import patch
from rapidtinyrec.audit_bf16 import evaluate_phases
from rapidtinyrec.bf16_protocol import PROTOCOL_SHA256
from rapidtinyrec.bf16_saved_stages import validate_saved_stages


class SavedAuditTests(unittest.TestCase):
    def fixture(self):
        start=dict(schema='bf16-pilot-work-v1',job_id='8040768',protocol_sha256=PROTOCOL_SHA256,
                   started_monotonic=100.,deadline_monotonic=1720.)
        gen=dict(name='generation',started_monotonic=101.,ended_monotonic=1200.,timeout_s=1500.,
                 within_deadline=True,child=dict(completed_normally=True))
        recovery=dict(name='recovery',started_monotonic=1201.,ended_monotonic=1205.,timeout_s=519.,
                      within_deadline=True,child=dict(completed_normally=True))
        return start,gen,recovery,dict(job_id='8040768')

    def test_saved_prerequisites_do_not_certify_original_audit(self):
        result=validate_saved_stages(*self.fixture())
        self.assertFalse(result['original_audit_certified'])
        self.assertEqual(result['original_job_id'],'8040768')

    def test_rejects_failed_or_overlapping_stages(self):
        for index,key,value in ((1,'within_deadline',False),(2,'started_monotonic',1199.),
                                (1,'name','audit'),(2,'ended_monotonic',1721.)):
            args=list(self.fixture());args[index][key]=value
            with self.assertRaises(AssertionError):validate_saved_stages(*args)
        args=list(self.fixture());args[1]['child']['completed_normally']=False
        with self.assertRaises(AssertionError):validate_saved_stages(*args)

    def test_deadlines_cannot_be_extended_or_nonfinite(self):
        for index,key,value in ((0,'deadline_monotonic',1800.),(1,'timeout_s',1600.),
                                (2,'timeout_s',520.),(1,'started_monotonic',float('nan')),
                                (2,'ended_monotonic',float('inf')),(1,'timeout_s',True)):
            args=list(self.fixture());args[index][key]=value
            with self.assertRaises(AssertionError):validate_saved_stages(*args)

    def test_original_identity_is_required(self):
        for key,value in (('job_id','other'),('protocol_sha256','other')):
            args=list(self.fixture());args[0][key]=value
            with self.assertRaises(AssertionError):validate_saved_stages(*args)

    def test_numerical_evaluator_rejects_local_or_old_allocation_before_io(self):
        # No model, file IO or bootstrap can begin under a stale generation ID.
        identity=dict(scheduler_fields=dict(JobId='8040768',JobState='RUNNING'))
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(AssertionError,'Allocated quality audit only'):
                evaluate_phases(None,None,None,None,wrapper=None,allocation_identity=identity)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'9999999'},clear=True):
            with self.assertRaisesRegex(AssertionError,'Current audit allocation required'):
                evaluate_phases(None,None,None,None,wrapper=None,allocation_identity=identity)
        identity['scheduler_fields'].update(JobId='9999999',JobState='COMPLETED')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'9999999'},clear=True):
            with self.assertRaisesRegex(AssertionError,'Current audit allocation required'):
                evaluate_phases(None,None,None,None,wrapper=None,allocation_identity=identity)

if __name__=='__main__':unittest.main()
