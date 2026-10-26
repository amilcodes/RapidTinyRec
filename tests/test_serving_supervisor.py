import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.output=self.root/'supervision'

    def run_child(self, source, seconds=3, observer=None, started=None):
        start=time.perf_counter_ns() if started is None else started
        return supervisor._run_group([sys.executable,'-B','-c',source],self.root,dict(os.environ),self.output,
            started_ns=start,deadline_ns=start+int(seconds*1e9),
            observe_identity=observer or (lambda pid:dict(pid=pid,test_only=True)))

    def test_success_retains_stdout_launch_and_exit_without_advancement(self):
        result=self.run_child("print('raw output')")
        # Darwin may return EPERM for a group containing only its zombie leader.
        # That must remain a failed cleanup, while retaining/reaping the exit.
        if sys.platform=='darwin' and result['error']:
            self.assertIn('PermissionError',result['error']);self.assertFalse(result['complete'])
        else:self.assertTrue(result['complete'])
        self.assertFalse(result['advancement_ready'])
        self.assertEqual(result['returncode'],0);self.assertFalse(result['deadline_exceeded'])
        self.assertIn('raw output',(self.output/'coordinator.log').read_text())
        launch=json.loads((self.output/'coordinator-launch.json').read_text())
        self.assertEqual(launch['pid'],result['child_pid'])
        self.assertLess(launch['launched_ns'],result['exit_observed_ns'])
        with self.assertRaises(FileExistsError):self.run_child('pass')

    def test_failed_coordinator_retains_nonzero_status(self):
        result=self.run_child("import sys; print('preflight failure'); sys.exit(7)")
        self.assertFalse(result['complete']);self.assertEqual(result['returncode'],7)
        self.assertFalse(result['deadline_exceeded'])

    def test_stalled_preflight_is_killed_without_cooperative_checks(self):
        result=self.run_child("import time; print('before preflight',flush=True); time.sleep(60)",seconds=.2)
        self.assertFalse(result['complete']);self.assertTrue(result['deadline_exceeded'])
        self.assertEqual(result['returncode'],-signal.SIGKILL)
        self.assertTrue(result['watchdog']['signal_sent'])
        self.assertLess((result['finished_ns']-result['global_started_ns'])/1e9,3)

    def test_watchdog_still_runs_during_parent_observation(self):
        observed=[]
        def slow_observer(pid):
            time.sleep(.5)
            observed.append(time.perf_counter_ns())
            return dict(pid=pid,test_only=True)
        result=self.run_child('import time; time.sleep(60)',seconds=.15,observer=slow_observer)
        self.assertTrue(result['deadline_exceeded'])
        self.assertLess(result['watchdog']['signal_ns'],observed[0])
        self.assertEqual(result['returncode'],-signal.SIGKILL)

    def test_observation_failure_cleans_up_unreaped_child(self):
        def fail(pid):raise ValueError('birth identity missing')
        result=self.run_child('import time; time.sleep(60)',observer=fail)
        self.assertFalse(result['complete']);self.assertIn('birth identity missing',result['error'])
        self.assertEqual(result['returncode'],-signal.SIGKILL)

    def test_expired_absolute_deadline_never_starts_a_new_process(self):
        with patch.object(subprocess,'Popen',side_effect=AssertionError('must not launch')):
            result=self.run_child('pass',seconds=.01,started=time.perf_counter_ns()-1_000_000_000)
        self.assertFalse(result['complete']);self.assertIsNone(result['child_pid'])
        self.assertTrue(result['deadline_exceeded'])

    def test_descendant_cannot_outlive_successful_group_leader(self):
        # A descendant would write after its leader exits if group cleanup failed.
        source="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',\"import time,pathlib; time.sleep(.8); pathlib.Path('escaped').write_text('bad')\"]); print(p.pid,flush=True)"
        result=self.run_child(source)
        self.assertTrue(result['complete']);self.assertTrue(result['group_cleanup_signal_sent'])
        time.sleep(1)
        self.assertFalse((self.root/'escaped').exists())

    def test_fixed_production_ceiling_cannot_be_extended(self):
        with patch.object(supervisor,'_run_group',return_value={}) as run:
            supervisor.supervise(['python'],self.root,{},self.output,global_started_ns=123)
        self.assertEqual(run.call_args.kwargs['deadline_ns'],123+1620*10**9)


if __name__=='__main__':unittest.main()
