"""Synthetic controller deadlines, failed-child closure and CLI boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from rapidtinyrec.buffer_reuse_runner import (allowance,commands_for,save,run_stages,validate_audit_start,PROTOCOL_SHA256)

ROOT=Path(__file__).resolve().parents[1]

class RunnerTests(unittest.TestCase):
    def fixture(self,out):
        record=dict(run_id='test',job_id='123',remote_dir=str(out.parent))
        commands=commands_for(out.parent/'code',out,Path('/tmp/wave-fixture'),sys.executable)
        save(out/'reuse-work-start.json',dict(schema='released-buffer-reuse-work-v1',job_id='123',run_id='test',
            protocol_sha256=PROTOCOL_SHA256,started_monotonic=100.,deadline_monotonic=1720.))
        save(out/'log-wrapper-start.json',dict(local_directory='/tmp/wave-fixture',job_id='123'))
        return record,commands
    def execute(self,out,fail=None,*,exception=False,delays=(10.,1.,1.)):
        record,commands=self.fixture(out);clock=[101.];seen=[]
        def supervisor(command,env,timeout):
            name=next(k for k,v in commands.items() if v==command);seen.append(name)
            clock[0]+=delays[len(seen)-1]
            if name=='recovery':save(out/'log-recovery.json',dict(recovery_complete=fail!='recovery'))
            if name=='audit':
                validate_audit_start(out,record,clock[0])
                save(out/'buffer-reuse-audit.json',dict(execution_audit_passed=fail!='audit'))
            if fail==name and exception:raise OSError('synthetic failed launch')
            return dict(schema='precision-child-exit-v1',completed_normally=fail!=name,
                        returncode=1 if fail==name else 0,reason='exited',signal=None,elapsed_s=delays[len(seen)-1])
        report=run_stages(out,record,commands,{},100.,supervisor=supervisor,clock=lambda:clock[0])
        return report,seen,record
    def test_normal_three_stage_chain_uses_shared_and_short_audit_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs';out.mkdir();report,seen,record=self.execute(out)
            self.assertTrue(report['completed_normally']);self.assertEqual(seen,['generation','recovery','audit'])
            budget=validate_audit_start(out,record,112.)
            self.assertEqual(budget['effective_deadline_monotonic'],232.)
            with self.assertRaises(ValueError):validate_audit_start(out,record,232.)
    def test_generation_failure_still_preserves_logs_and_skips_audit(self):
        for exception in (False,True):
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)/'outputs';out.mkdir();report,seen,_=self.execute(out,'generation',exception=exception)
                self.assertFalse(report['completed_normally']);self.assertEqual(seen,['generation','recovery'])
                self.assertIn('audit_skipped',report);self.assertFalse((out/'buffer-reuse-audit.json').exists())
                wrapper=json.loads((out/'log-wrapper-exit.json').read_text())
                self.assertFalse(wrapper['child']['completed_normally']);self.assertTrue(wrapper['recovery']['recovery_complete'])
    def test_recovery_failure_cannot_launch_audit(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs';out.mkdir();report,seen,_=self.execute(out,'recovery')
            self.assertEqual(seen,['generation','recovery']);self.assertFalse(report['completed_normally'])
            self.assertFalse(json.loads((out/'log-wrapper-exit.json').read_text())['recovery']['recovery_complete'])
    def test_failed_audit_remains_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs';out.mkdir();report,seen,_=self.execute(out,'audit')
            self.assertEqual(len(seen),3);self.assertFalse(report['completed_normally'])
            self.assertFalse(json.loads((out/'buffer-reuse-audit.json').read_text())['execution_audit_passed'])
    def test_generation_overrun_recovers_but_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs';out.mkdir();report,seen,_=self.execute(out,delays=(1501.,1.,1.))
            self.assertEqual(seen,['generation','recovery']);self.assertFalse(report['completed_normally'])
            self.assertFalse(report['stages'][0]['within_deadline'])
    def test_no_budget_means_no_additional_work(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'outputs';out.mkdir();report,seen,_=self.execute(out,delays=(1620.,1.,1.))
            self.assertEqual(seen,['generation']);self.assertEqual(report['deadline_before'],'recovery')
            self.assertFalse(report['completed_normally'])
    def test_original_stage_identity_order_commands_and_budgets_are_audited(self):
        changes=[('reuse-work-start.json','run_id','other'),('reuse-generation-exit.json','job_id','other'),
                 ('reuse-recovery-exit.json','started_monotonic',100.),('reuse-audit-start.json','timeout_s',121.),
                 ('reuse-audit-start.json','command',['true'])]
        for name,key,value in changes:
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)/'outputs';out.mkdir();_,_,record=self.execute(out);p=out/name;data=json.loads(p.read_text());data[key]=value;p.write_text(json.dumps(data))
                with self.subTest(name=name,key=key),self.assertRaises(ValueError):validate_audit_start(out,record,113.)
    def test_budget_is_finite_and_inclusive(self):
        self.assertEqual(allowance('generation',100,101),1500)
        self.assertEqual(allowance('audit',100,1650),70)
        self.assertEqual(allowance('recovery',100,1720),0)
        for bad in (float('nan'),float('inf'),True,99):
            with self.assertRaises(ValueError):allowance('audit',100,bad)
    def test_cli_refuses_local_and_duplicate_original_work(self):
        env={k:v for k,v in os.environ.items() if k not in ('SLURM_JOB_ID','OSCAR_RUN_DIR')}
        command=[sys.executable,'-m','rapidtinyrec.buffer_reuse_runner','run']
        p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('Allocated buffer-reuse',p.stderr)
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);out=run/'outputs';out.mkdir();marker=out/'reuse-work-start.json';marker.write_bytes(b'original')
            p=subprocess.run(command,cwd=ROOT,env=dict(env,SLURM_JOB_ID='123',OSCAR_RUN_DIR=d),capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('never restart generation',p.stderr)
            self.assertEqual(marker.read_bytes(),b'original');self.assertFalse((out/'reuse-work-exit.json').exists())

if __name__=='__main__':unittest.main()
