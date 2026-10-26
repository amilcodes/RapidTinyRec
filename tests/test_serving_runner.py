import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_runner as a
from rapidtinyrec import serving_supervisor, serving_coordinator, serving_run_audit
from rapidtinyrec.serving_setup_audit import read_unique


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name).resolve()
        (self.run/'outputs/coordinator').mkdir(parents=True)
        self.args=['--run-root',str(self.run),'--release-root','/release','--historical','/history']
        p=patch.dict(os.environ,OSCAR_RUN_DIR=str(self.run),OSCAR_OUTPUT_DIR=str(self.run/'outputs'),SLURM_JOB_ID='12345')
        p.start();self.addCleanup(p.stop)
        p=patch.object(sys,'dont_write_bytecode',True);p.start();self.addCleanup(p.stop)

    def test_outer_uses_entry_clock_and_exact_supervised_child_command(self):
        with patch.object(serving_supervisor,'supervise',return_value={'complete':True}) as supervisor, patch.object(a,'run_coordinator',side_effect=AssertionError('unsupervised child')):
            self.assertEqual(a.main(self.args),0)
        args,kw=supervisor.call_args
        self.assertEqual(args[0],[sys.executable,'-B','-m','rapidtinyrec.serving_runner','--coordinator',*self.args])
        self.assertEqual(args[1],self.run/'code');self.assertEqual(args[3],self.run/'outputs/supervision')
        self.assertEqual(kw,{'global_started_ns':a._ENTRY_STARTED_NS})

    def test_outer_failure_cannot_return_success(self):
        with patch.object(serving_supervisor,'supervise',return_value={'complete':False}):self.assertEqual(a.main(self.args),1)

    def test_child_binds_mandatory_disk_callback_and_preserves_negative_result(self):
        def coordinate(run,release,history,*,final_auditor):
            self.assertIs(final_auditor,a.allocated_auditor)
            self.assertFalse(final_auditor(run,history)['evidence_gates_passed'])
        with patch.object(serving_coordinator,'coordinate',side_effect=coordinate), patch.object(serving_run_audit,'audit_pre_exit',return_value={'evidence_gates_passed':False}) as audit, patch.object(serving_supervisor,'supervise',side_effect=AssertionError('nested supervisor')):
            self.assertEqual(a.main(['--coordinator',*self.args]),0)
            audit.assert_called_once_with(self.run,Path('/history'))
        self.assertFalse((self.run/'outputs/coordinator/audit-failure.json').exists())

    def test_malformed_audit_preserves_failure_without_replacing_evidence(self):
        with patch.object(serving_run_audit,'audit_pre_exit',side_effect=ValueError('raw trace missing')):
            with self.assertRaisesRegex(ValueError,'raw trace missing'):a.allocated_auditor(self.run,Path('/history'))
            path=self.run/'outputs/coordinator/audit-failure.json';saved=path.read_bytes()
            self.assertFalse(read_unique(path)['evidence_gates_passed'])
            with self.assertRaises(FileExistsError):a.allocated_auditor(self.run,Path('/history'))
            self.assertEqual(path.read_bytes(),saved)

    def test_child_preflight_failure_is_retained_and_raised(self):
        with patch.object(serving_coordinator,'coordinate',side_effect=ValueError('allocation mismatch')):
            with self.assertRaisesRegex(ValueError,'allocation mismatch'):a.main(['--coordinator',*self.args])
        self.assertIn('allocation mismatch',read_unique(self.run/'outputs/coordinator-failure.json')['error'])

    def test_bad_environment_fails_before_supervisor_or_coordinator(self):
        with patch.object(a,'run_supervised',side_effect=AssertionError('launched')),patch.object(a,'run_coordinator',side_effect=AssertionError('loaded')):
            for key,value in [('SLURM_JOB_ID',''),('OSCAR_RUN_DIR','/elsewhere'),('OSCAR_OUTPUT_DIR','/other')]:
                with patch.dict(os.environ,{key:value}),self.assertRaises(ValueError):a.main(self.args)
            with patch.object(sys,'dont_write_bytecode',False),self.assertRaisesRegex(ValueError,'requires -B'):a.main(self.args)

    def test_optimized_cli_fails_without_loading_torch(self):
        command=[sys.executable,'-B','-O','-m','rapidtinyrec.serving_runner',*self.args]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('nonoptimized Python',result.stderr)
        self.assertFalse((self.run/'outputs/supervision').exists())

    def test_help_does_not_import_execution_or_torch(self):
        code="import sys;from rapidtinyrec.serving_runner import main\ntry:main(['--help'])\nexcept SystemExit as e:assert e.code==0\nassert 'torch' not in sys.modules\nassert 'rapidtinyrec.serving_coordinator' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
