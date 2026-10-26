import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from rapidtinyrec.precision_log_runner import supervise, recover_logs
from rapidtinyrec.precision_log_staging import LogStage, LOG_NAMES, save_record


class SupervisorTests(unittest.TestCase):
    def test_normal_exit_and_failed_exit(self):
        before = signal.getsignal(signal.SIGTERM)
        result = supervise([sys.executable, '-c', 'pass'], dict(os.environ), 5)
        self.assertTrue(result['completed_normally'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        result = supervise([sys.executable, '-c', 'raise SystemExit(7)'], dict(os.environ), 5)
        self.assertEqual(result['returncode'], 7)
        self.assertFalse(result['completed_normally'])

    def test_deadline_kills_child_ignoring_term_and_retains_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'written'
            code = 'import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path('+repr(str(path))+').write_bytes(b"partial");time.sleep(60)'
            result = supervise([sys.executable, '-c', code], dict(os.environ), .5, .1)
            self.assertEqual(result['reason'], 'deadline')
            self.assertFalse(result['completed_normally'])
            self.assertEqual(result['returncode'], -signal.SIGKILL)
            self.assertEqual(path.read_bytes(), b'partial')

    def test_wrapper_signal_is_forwarded_without_losing_control(self):
        code = 'import os,signal,time;os.kill(os.getppid(),signal.SIGTERM);time.sleep(60)'
        result = supervise([sys.executable, '-c', code], dict(os.environ), 5, .2)
        self.assertEqual(result['reason'], 'signal')
        self.assertEqual(result['signal'], signal.SIGTERM)
        self.assertFalse(result['completed_normally'])

    def test_spawn_failure_restores_handlers(self):
        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(FileNotFoundError): supervise(['/no/such/program'], dict(os.environ), 5)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve(); self.local = self.base/'local'; self.local.mkdir()
        self.out = self.base/'out'; self.out.mkdir(); self.phase = self.out/'phase_1'; self.phase.mkdir()
        self.fs = {'mount': '/', 'filesystem': 'tmpfs'}
        self.stage = LogStage(self.local, self.phase, self.fs)
        self.location = self.phase/'log-staging-location.json'
        save_record(self.location, {'schema':'precision-log-location-v1', 'local_directory':str(self.stage.root),
                    'output_directory':str(self.phase), 'filesystem':self.fs})

    def test_partial_log_is_preserved_after_child_exit(self):
        (self.stage.root/LOG_NAMES[0]).write_bytes(b'partial')
        result = recover_logs(self.out, self.local, self.fs)
        self.assertTrue(result['recovery_complete'])
        self.assertFalse(result['locations'][0]['publication']['complete'])
        self.assertEqual((self.phase/LOG_NAMES[0]).read_bytes(), b'partial')

    def test_existing_conflict_retains_local_source(self):
        (self.stage.root/LOG_NAMES[0]).write_bytes(b'new')
        (self.phase/LOG_NAMES[0]).write_bytes(b'existing')
        result = recover_logs(self.out, self.local, self.fs)
        self.assertFalse(result['recovery_complete'])
        self.assertEqual((self.stage.root/LOG_NAMES[0]).read_bytes(), b'new')
        self.assertEqual((self.phase/LOG_NAMES[0]).read_bytes(), b'existing')

    def test_location_cannot_escape_private_directory(self):
        data = json.loads(self.location.read_text()); data['local_directory'] = str(self.base)
        save_record(self.location, data)
        result = recover_logs(self.out, self.local, self.fs)
        self.assertFalse(result['recovery_complete'])
        self.assertIn('outside private', result['locations'][0]['error'])

    def test_unmapped_local_entries_block_cleanup_success(self):
        self.location.unlink()
        (self.stage.root/LOG_NAMES[0]).write_bytes(b'unknown')
        result = recover_logs(self.out, self.local, self.fs)
        self.assertFalse(result['recovery_complete'])
        self.assertTrue(self.stage.root.exists())


if __name__ == '__main__': unittest.main()
