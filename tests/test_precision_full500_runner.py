import json
from pathlib import Path
import tempfile
import unittest

from rapidtinyrec.precision_full500 import PROTOCOL_SHA256
from rapidtinyrec.precision_full500_runner import (save,stage_allowance,run_stages,
    validate_audit_start,reconcile_work,digest)

class Full500RunnerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.started=100.;self.now=100.;self.calls=[];self.durations={'generation':1470.,'recovery':10.,'audit':100.}
        self.fail=None;self.error=None
        self.record={'job_id':'fixture','run_id':'fixture-run'}
        save(self.root/'allocated-run-record.json',self.record)
        save(self.root/'full500-work-start.json',dict(schema='precision-full500-work-start-v1',job_id='fixture',
            protocol_sha256=PROTOCOL_SHA256,started_monotonic=100.,deadline_monotonic=1720.,global_work_deadline_s=1620))
    def launch(self,command,env,timeout):
        name=command[0];self.calls.append((name,timeout));self.now+=self.durations[name]
        if self.error==name:raise RuntimeError('fixture failure')
        if name=='recovery':save(self.root/'log-recovery.json',{'recovery_complete':self.fail!='recovery','locations':[{}, {}, {}, {}]})
        if name=='audit':save(self.root/'full500-audit.json',dict(full500_execution_audit_passed=self.fail!='audit',
            job_id='fixture',run_id='fixture-run',allocated_record_sha256=digest(self.root/'allocated-run-record.json')))
        return dict(completed_normally=self.fail!=name,reason='exited',returncode=1 if self.fail==name else 0)
    def run_fixture(self):
        return run_stages(self.root,self.started,{n:[n] for n in ('generation','recovery','audit')},{},clock=lambda:self.now,launch=self.launch)
    def test_recovery_precedes_audit_and_complete_result_reconciles(self):
        result=self.run_fixture();self.assertTrue(result['completed_normally'])
        self.assertEqual(self.calls,[('generation',1500),('recovery',150),('audit',120)])
        self.assertTrue(reconcile_work(self.root,self.record)['passed'])
        self.assertFalse(reconcile_work(self.root,self.record)['bootstrap_executed'])
    def test_identity_time_is_in_global_budget(self):
        self.now=230.;self.durations.update(generation=1450,recovery=10,audit=20)
        self.assertTrue(self.run_fixture()['completed_normally'])
        self.assertEqual(self.calls,[('generation',1490),('recovery',40),('audit',30)])
    def test_audit_receives_remaining_time_after_recovery(self):
        self.durations.update(generation=1490,recovery=40,audit=80)
        self.assertTrue(self.run_fixture()['completed_normally'])
        self.assertEqual(self.calls[-1],('audit',90))
    def test_generation_failure_still_recovers_but_never_audits(self):
        self.fail='generation';self.assertFalse(self.run_fixture()['completed_normally'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
        self.assertTrue((self.root/'log-wrapper-exit.json').exists())
        self.assertFalse((self.root/'full500-audit-start.json').exists())
    def test_recovery_failure_never_audits(self):
        self.fail='recovery';self.assertFalse(self.run_fixture()['completed_normally'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
    def test_global_deadline_exhaustion_never_starts_next_stage(self):
        self.now=1720;result=self.run_fixture()
        self.assertEqual(self.calls,[]);self.assertEqual(result['deadline_exhausted_before'],'generation')
        self.assertFalse(result['completed_normally'])
    def test_audit_failure_retains_output_and_is_not_success(self):
        self.fail='audit';result=self.run_fixture();self.assertFalse(result['completed_normally'])
        self.assertEqual(result['audit_sha256'],digest(self.root/'full500-audit.json'))
        with self.assertRaises(AssertionError):reconcile_work(self.root,self.record)
    def test_termination_grace_cannot_turn_late_exit_into_success(self):
        self.durations.update(generation=1490,recovery=10,audit=121)
        self.assertFalse(self.run_fixture()['completed_normally'])
        with self.assertRaises(AssertionError):reconcile_work(self.root,self.record)
    def test_late_generation_exit_cannot_start_audit(self):
        self.durations.update(generation=1501,recovery=1)
        self.assertFalse(self.run_fixture()['completed_normally'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
    def test_supervisor_error_preserves_failure_and_does_not_audit(self):
        self.error='recovery'
        with self.assertRaises(RuntimeError):self.run_fixture()
        result=json.loads((self.root/'full500-work-exit.json').read_text())
        self.assertFalse(result['completed_normally']);self.assertIn('fixture failure',result['traceback'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
    def test_changed_order_deadline_and_result_bytes_rejected(self):
        self.run_fixture()
        audit_path=self.root/'full500-audit-start.json';original=audit_path.read_bytes();audit=json.loads(original)
        for key,val in [('started_monotonic',101),('timeout_s',121)]:
            changed=dict(audit,**{key:val});audit_path.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError):validate_audit_start(self.root,self.record)
        audit_path.write_bytes(original)
        with self.assertRaises(AssertionError):validate_audit_start(self.root,self.record,now=1721)
        result=self.root/'full500-audit.json';result.write_bytes(result.read_bytes()+b' ')
        with self.assertRaises(AssertionError):reconcile_work(self.root,self.record)
    def test_existing_work_records_cannot_be_overwritten(self):
        self.run_fixture();before=(self.root/'full500-work-exit.json').read_bytes()
        with self.assertRaises(FileExistsError):self.run_fixture()
        self.assertEqual(before,(self.root/'full500-work-exit.json').read_bytes())

if __name__=='__main__':unittest.main()
