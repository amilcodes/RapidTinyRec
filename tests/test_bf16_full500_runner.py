import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec.bf16_full500 import PROTOCOL_SHA256
from rapidtinyrec.bf16_full500_runner import (save,stage_allowance,run_stages,
    validate_audit_start,reconcile_work,digest)

class Bf16Full500RunnerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.started=100.;self.now=100.;self.calls=[];self.durations={'generation':1470.,'recovery':10.,'audit':100.}
        self.fail=None;self.error=None
        self.record={'job_id':'fixture','run_id':'fixture-run'}
        save(self.root/'allocated-run-record.json',self.record)
        save(self.root/'bf16-full500-work-start.json',dict(schema='bf16-full500-work-start-v1',job_id='fixture',
            protocol_sha256=PROTOCOL_SHA256,started_monotonic=100.,deadline_monotonic=1720.,global_work_deadline_s=1620))
    def launch(self,command,env,timeout):
        name=command[0];self.calls.append((name,timeout));self.now+=self.durations[name]
        if self.error==name:raise RuntimeError('fixture failure')
        if name=='recovery':save(self.root/'log-recovery.json',{'recovery_complete':self.fail!='recovery','locations':[{}, {}, {}, {}]})
        if name=='audit':save(self.root/'bf16-full500-audit.json',dict(schema='bf16-vs-tf32-full500-audit-v1',protocol_sha256=PROTOCOL_SHA256,execution_audit_passed=self.fail!='audit',
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
    def test_start_record_publication_consumes_stage_budget(self):
        def slow_save(path,value):
            save(path,value)
            if path.name=='bf16-full500-generation-start.json':self.now+=20
        self.durations.update(generation=1460,recovery=10,audit=100)
        with patch('rapidtinyrec.bf16_full500_runner.save',side_effect=slow_save):
            self.assertTrue(self.run_fixture()['completed_normally'])
        self.assertEqual(self.calls[0],('generation',1480))
    def test_expired_start_publication_never_launches_child(self):
        def slow_save(path,value):
            save(path,value)
            if path.name=='bf16-full500-generation-start.json':self.now+=1620
        with patch('rapidtinyrec.bf16_full500_runner.save',side_effect=slow_save):
            self.assertFalse(self.run_fixture()['completed_normally'])
        self.assertEqual(self.calls,[])
    def test_audit_receives_remaining_time_after_recovery(self):
        self.durations.update(generation=1490,recovery=40,audit=80)
        self.assertTrue(self.run_fixture()['completed_normally'])
        self.assertEqual(self.calls[-1],('audit',90))
    def test_generation_failure_still_recovers_but_never_audits(self):
        self.fail='generation';self.assertFalse(self.run_fixture()['completed_normally'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
        self.assertTrue((self.root/'log-wrapper-exit.json').exists())
        self.assertFalse((self.root/'bf16-full500-audit-start.json').exists())
    def test_recovery_failure_never_audits(self):
        self.fail='recovery';self.assertFalse(self.run_fixture()['completed_normally'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
    def test_global_deadline_exhaustion_never_starts_next_stage(self):
        self.now=1720;result=self.run_fixture()
        self.assertEqual(self.calls,[]);self.assertEqual(result['deadline_exhausted_before'],'generation')
        self.assertFalse(result['completed_normally'])
    def test_audit_failure_retains_output_and_is_not_success(self):
        self.fail='audit';result=self.run_fixture();self.assertFalse(result['completed_normally'])
        self.assertEqual(result['audit_sha256'],digest(self.root/'bf16-full500-audit.json'))
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
        result=json.loads((self.root/'bf16-full500-work-exit.json').read_text())
        self.assertFalse(result['completed_normally']);self.assertIn('fixture failure',result['traceback'])
        self.assertEqual([n for n,t in self.calls],['generation','recovery'])
    def test_changed_order_deadline_and_result_bytes_rejected(self):
        self.run_fixture()
        audit_path=self.root/'bf16-full500-audit-start.json';original=audit_path.read_bytes();audit=json.loads(original)
        for key,val in [('started_monotonic',101),('timeout_s',121)]:
            changed=dict(audit,**{key:val});audit_path.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError):validate_audit_start(self.root,self.record)
        audit_path.write_bytes(original)
        with self.assertRaises(AssertionError):validate_audit_start(self.root,self.record,now=1721)
        result=self.root/'bf16-full500-audit.json';result.write_bytes(result.read_bytes()+b' ')
        with self.assertRaises(AssertionError):reconcile_work(self.root,self.record)
    def test_existing_work_records_cannot_be_overwritten(self):
        self.run_fixture();before=(self.root/'bf16-full500-work-exit.json').read_bytes()
        with self.assertRaises(FileExistsError):self.run_fixture()
        self.assertEqual(before,(self.root/'bf16-full500-work-exit.json').read_bytes())



class Bf16Full500SubprocessTests(unittest.TestCase):
    def test_real_supervised_children_success_and_generation_failure(self):
        import hashlib,os,sys,time
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);started=time.monotonic()
                save(root/'allocated-run-record.json',dict(job_id='fixture',run_id='synthetic'))
                save(root/'bf16-full500-work-start.json',dict(schema='bf16-full500-work-start-v1',job_id='fixture',protocol_sha256=PROTOCOL_SHA256,started_monotonic=started,deadline_monotonic=started+1620,global_work_deadline_s=1620))
                child=root/'child.py'
                child.write_text("""import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);stage=sys.argv[2]
if stage=='generation':
    (root/'partial-generation.txt').write_text('synthetic saved output')
    raise SystemExit(int(sys.argv[3]))
if stage=='recovery':
    (root/'log-recovery.json').write_text(json.dumps(dict(recovery_complete=True,locations=[{}, {}, {}, {}])))
if stage=='audit':
    (root/'bf16-full500-audit.json').write_text(json.dumps(dict(schema='bf16-vs-tf32-full500-audit-v1',protocol_sha256='c2bd251a8b9c34e23d2a14a85979fea8778d175485241fb650e04188e99f17cc',execution_audit_passed=True,job_id='fixture',run_id='synthetic',allocated_record_sha256=hashlib.sha256((root/'allocated-run-record.json').read_bytes()).hexdigest())))
""")
                commands={name:[sys.executable,str(child),str(root),name,str(int(failure))] for name in ('generation','recovery','audit')}
                result=run_stages(root,started,commands,dict(os.environ))
                self.assertEqual(result['completed_normally'],not failure)
                self.assertTrue((root/'partial-generation.txt').exists())
                self.assertTrue((root/'log-wrapper-exit.json').exists())
                self.assertEqual((root/'bf16-full500-audit.json').exists(),not failure)
                if not failure:self.assertTrue(reconcile_work(root,dict(job_id='fixture',run_id='synthetic'))['passed'])

    def test_allocated_cli_fails_locally_and_retains_original_failure(self):
        import os,subprocess,sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'audit.json';env=dict(os.environ)
            env.pop('SLURM_JOB_ID',None)
            cmd=[sys.executable,'-m','rapidtinyrec.audit_bf16_full500','allocated']
            for name in ('root','historical','snapshot','run-root','release-root'):cmd+=['--'+name,str(root)]
            cmd+=['--output',str(output)]
            result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,1)
            saved=output.read_bytes();record=json.loads(saved)
            self.assertIn('requires allocated compute',record['error']);self.assertFalse(record['execution_audit_passed'])
            self.assertEqual(subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=10).returncode,1)
            self.assertEqual(output.read_bytes(),saved)

    def test_nonfinite_or_reversed_clock_rejected(self):
        for start,now in ((0,float('nan')),(float('inf'),2),(True,2),(10,9)):
            with self.assertRaises(AssertionError):stage_allowance('generation',start,now)

if __name__=='__main__':unittest.main()
