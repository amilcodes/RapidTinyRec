import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_terminal_audit as a
import test_serving_run_audit as run_tests
from test_serving_disk_audit import read, write


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class TerminalTests(unittest.TestCase):
    def setUp(self):
        fixture=run_tests.RunAuditTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.run=fixture.run
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.external=Path(temp.name)
        fixture.history.rename(self.external/'history');fixture.history=self.external/'history';self.history=fixture.history
        protocol=run_tests.a.load_protocol(None)
        p=patch.object(a,'load_protocol',return_value=protocol);p.start();self.addCleanup(p.stop)
        self.report=fixture.call()
        write(self.run/'outputs/coordinator/final-audit.json',self.report)
        write(self.run/'outputs/coordinator/exit.json',dict(identity=fixture.parent,finished_ns=4000020,
            phases_completed=[0,1,2,3],complete=True,error=None,advancement_ready=False,
            audit_published_ns=4000010,audit_sha256=digest(self.run/'outputs/coordinator/final-audit.json')))
        write(self.run/'outputs/supervision/supervisor-exit.json',dict(schema='serving-supervisor-exit-v1',
            global_started_ns=1,deadline_ns=1+1620*10**9,exit_observed_ns=4000030,finished_ns=4000040,
            child_pid=100,returncode=0,deadline_exceeded=False,watchdog={},group_cleanup_signal_sent=True,
            complete=True,error=None,advancement_ready=False))
        self.actual=self.external/'submitted.json';write(self.actual,fixture.allocation.records()[1])
        self.freeze=self.external/'freeze.json';self.refreeze()
        # Terminal reconciliation must never re-execute any allocated audit.
        for module,name in [('rapidtinyrec.serving_run_audit','audit_pre_exit'),('rapidtinyrec.serving_disk_audit','audit_run')]:
            p=patch(module+'.'+name,side_effect=AssertionError('Numerical audit rerun'));p.start();self.addCleanup(p.stop)

    def refreeze(self):
        report=read(self.run/'outputs/coordinator/final-audit.json')
        inputs=copy.deepcopy(report['input_sha256'])
        for key in a.TERMINAL:inputs[key]=digest(self.run/key.removeprefix('run/'))
        actual=read(self.actual)
        write(self.freeze,dict(schema='serving-terminal-freeze-v1',run_id=actual['run_id'],job_id=actual['job_id'],
                              submitted_record_sha256=digest(self.actual),input_sha256=inputs))

    def call(self):return a.reconcile_terminal(self.run,self.history,self.actual,self.freeze)

    def change(self,relative,**values):
        path=self.run/relative;row=read(path);row.update(values);write(path,row);self.refreeze()

    def test_terminal_metadata_only_without_phase_audit(self):
        result=self.call();self.assertTrue(result['metadata_reconciled'])
        self.assertTrue(result['allocation_binding']['submitted_record_reconciled'])
        self.assertTrue(result['allocated_evidence_gates_passed']);self.assertFalse(result['advancement_ready'])
        self.assertFalse(result['numerical_audit_executed']);self.assertFalse(result['model_executed'])
        self.assertEqual(result['terminal_stdout_sha256'],digest(self.run/'outputs/supervision/coordinator.log'))

    def test_changed_pre_exit_or_deferred_stdout_bytes_rejected(self):
        for relative in ('outputs/phase_1/execution/profile/trace.json','outputs/supervision/coordinator.log'):
            path=self.run/relative;original=path.read_bytes();path.write_bytes(original+b' ')
            with self.subTest(relative=relative),self.assertRaisesRegex(ValueError,'Frozen input bytes'):self.call()
            path.write_bytes(original)

    def test_actual_submitted_record_binding_cannot_be_fabricated(self):
        row=read(self.actual);row['job_id']='99999';write(self.actual,row);self.refreeze()
        with self.assertRaisesRegex(ValueError,'Submission states/job'):self.call()

    def test_allocated_report_publication_hash_and_clock_required(self):
        self.change('outputs/coordinator/exit.json',audit_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'publication hash'):self.call()
        self.change('outputs/coordinator/exit.json',audit_sha256=digest(self.run/'outputs/coordinator/final-audit.json'),audit_published_ns=2)
        with self.assertRaisesRegex(ValueError,'clocks'):self.call()

    def test_supervisor_timeout_and_wrong_process_never_pass(self):
        self.change('outputs/supervision/supervisor-exit.json',deadline_exceeded=True)
        with self.assertRaisesRegex(ValueError,'Supervisor terminal'):self.call()
        self.change('outputs/supervision/supervisor-exit.json',deadline_exceeded=False,child_pid=101)
        with self.assertRaisesRegex(ValueError,'Supervisor terminal'):self.call()

    def test_missing_input_and_extra_file_cannot_evade_inventory(self):
        path=self.run/'outputs/coordinator/final-audit.json';row=read(path)
        del row['input_sha256']['run/outputs/phase_0/lifecycle.json'];write(path,row);self.refreeze()
        with self.assertRaisesRegex(ValueError,'Complete terminal file inventory'):self.call()

    def test_failed_allocated_gate_stays_false_with_normal_exit(self):
        path=self.run/'outputs/coordinator/final-audit.json';row=read(path)
        row['gates']['performance']=False;row['phases']['gates']['performance']=False
        row['evidence_gates_passed']=False;row['phases']['evidence_gates_passed']=False;write(path,row)
        self.change('outputs/coordinator/exit.json',audit_sha256=digest(path))
        result=self.call();self.assertTrue(result['metadata_reconciled'])
        self.assertFalse(result['allocated_evidence_gates_passed']);self.assertFalse(result['gates']['performance'])

    def test_gate_flag_cannot_override_allocated_failed_gate(self):
        path=self.run/'outputs/coordinator/final-audit.json';row=read(path);row['gates']['performance']=False;write(path,row)
        self.change('outputs/coordinator/exit.json',audit_sha256=digest(path))
        with self.assertRaisesRegex(ValueError,'gate summary'):self.call()

    def test_frozen_path_traversal_rejected(self):
        path=self.run/'outputs/coordinator/final-audit.json';row=read(path);row['input_sha256']['run/../outside']='a'*64;write(path,row);self.refreeze()
        with self.assertRaisesRegex(ValueError,'Invalid frozen input path'):self.call()


if __name__=='__main__':unittest.main()
