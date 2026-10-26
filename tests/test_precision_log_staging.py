import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rapidtinyrec.precision_log_staging import LogStage, LOG_NAMES, filesystem_for, StagedCapture
import json


class LogStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.local = self.root/'local'; self.local.mkdir()
        self.out = self.root/'out'; self.out.mkdir()
        self.stage = LogStage(self.local, self.out, {'mount': str(self.local), 'filesystem': 'tmpfs'})

    def fill(self):
        for name in LOG_NAMES:
            (self.stage.root/name).write_bytes((name+'\n').encode()*100)

    def test_complete_bytes_and_safe_cleanup(self):
        self.fill()
        report = self.stage.publish(loggers_disabled=True, require_complete=True)
        self.assertTrue(report['complete'])
        for name in LOG_NAMES:
            self.assertEqual((self.out/name).read_bytes(), (self.stage.root/name).read_bytes())
        self.stage.cleanup(); self.assertFalse(self.stage.root.exists())

    def test_partial_exception_evidence_and_idempotent_retry(self):
        (self.stage.root/LOG_NAMES[0]).write_bytes(b'partial')
        report = self.stage.publish(loggers_disabled=True, require_complete=False)
        self.assertFalse(report['complete'])
        self.assertEqual(self.stage.publish(loggers_disabled=True, require_complete=False), report)
        self.assertEqual((self.out/LOG_NAMES[0]).read_bytes(), b'partial')

    def test_missing_and_empty_logs_fail_complete_publication(self):
        with self.assertRaises(FileNotFoundError):
            self.stage.publish(loggers_disabled=True, require_complete=True)
        self.fill(); (self.stage.root/LOG_NAMES[1]).write_bytes(b'')
        with self.assertRaises(ValueError):
            self.stage.publish(loggers_disabled=True, require_complete=True)
        self.assertFalse(any(self.out.iterdir()))

    def test_copy_failure_retains_source_and_retries(self):
        self.fill()
        with patch('rapidtinyrec.precision_log_staging.os.link', side_effect=OSError('full disk')):
            with self.assertRaises(OSError):
                self.stage.publish(loggers_disabled=True, require_complete=True)
        self.assertTrue(all((self.stage.root/n).exists() for n in LOG_NAMES))
        with self.assertRaises(RuntimeError): self.stage.cleanup()
        self.assertFalse(any(self.out.iterdir()))
        self.stage.publish(loggers_disabled=True, require_complete=True)

    def test_cannot_overwrite_evidence_or_copy_open_logger(self):
        self.fill(); (self.out/LOG_NAMES[1]).write_bytes(b'old evidence')
        with self.assertRaises(RuntimeError):
            self.stage.publish(loggers_disabled=False, require_complete=True)
        with self.assertRaises(FileExistsError):
            self.stage.publish(loggers_disabled=True, require_complete=True)
        self.assertEqual((self.out/LOG_NAMES[1]).read_bytes(), b'old evidence')
        self.assertFalse((self.out/LOG_NAMES[0]).exists())

    def test_symlink_and_unexpected_file_rejected(self):
        self.fill(); (self.stage.root/'unexpected').write_text('keep')
        with self.assertRaises(ValueError): self.stage.publish(loggers_disabled=True, require_complete=True)
        (self.stage.root/'unexpected').unlink()
        (self.out/LOG_NAMES[0]).symlink_to(self.stage.root/LOG_NAMES[0])
        with self.assertRaises(ValueError): self.stage.publish(loggers_disabled=True, require_complete=True)

    def test_corrupted_destination_prevents_cleanup(self):
        self.fill(); self.stage.publish(loggers_disabled=True, require_complete=True)
        (self.out/LOG_NAMES[0]).write_bytes(b'corruption')
        with self.assertRaises(OSError): self.stage.cleanup()
        self.assertTrue(self.stage.root.exists())

    def test_mount_selection_and_allocation_guard(self):
        mounts = '1 0 0:1 / / rw - ext4 disk rw\n2 1 0:2 / '+str(self.local)+' rw - nfs server rw\n'
        with self.assertRaises(ValueError): filesystem_for(self.local, mounts)
        self.assertEqual(filesystem_for(self.local, mounts.replace('nfs server', 'tmpfs tmpfs'))['filesystem'], 'tmpfs')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError): LogStage.for_allocation(self.local, self.out)


class StagedCaptureTests(unittest.TestCase):
    setUp = LogStageTests.setUp
    def logger(self, fail_disable=False):
        stage = self.stage
        class FakeLogger:
            active = False
            def begin(self, prefix):
                self.active = True
                for name in LOG_NAMES: (stage.root/name).write_bytes(b'API evidence')
                return {'available': True, 'cublas': LOG_NAMES[0], 'cublaslt': LOG_NAMES[1]}
            def end_verified(self):
                if fail_disable: raise RuntimeError('disable failed')
                self.active = False
        return FakeLogger()

    def test_success_disables_and_flushes_before_publication(self):
        logger = self.logger(); flushed = []
        def flush():
            self.assertFalse(logger.active); flushed.append(True)
        with StagedCapture(self.stage, logger, flush) as evidence:
            self.assertTrue(logger.active)
            self.assertTrue(evidence['available'])
            self.assertFalse((self.out/LOG_NAMES[0]).exists())
        self.assertEqual(flushed, [True])
        record = json.loads((self.out/'log-staging.json').read_text())
        self.assertTrue(record['measurement_allowed'])
        from rapidtinyrec.audit_precision import validate_log_staging
        validate_log_staging(self.out, {'api_log_staging': True})
        (self.out/LOG_NAMES[0]).write_bytes(b'corruption')
        with self.assertRaises(AssertionError):
            validate_log_staging(self.out, {'api_log_staging': True})
        self.assertTrue(self.stage.root.exists())

    def test_setup_exception_preserves_partial_logs_and_propagates(self):
        with self.assertRaisesRegex(ValueError, 'setup failed'):
            with StagedCapture(self.stage, self.logger(), lambda: None):
                raise ValueError('setup failed')
        record = json.loads((self.out/'log-staging.json').read_text())
        self.assertTrue(record['published'])
        self.assertFalse(record['measurement_allowed'])
        self.assertIn('setup failed', record['setup_error'])

    def test_disable_failure_blocks_publication_and_measurement(self):
        with self.assertRaisesRegex(RuntimeError, 'disable failed'):
            with StagedCapture(self.stage, self.logger(True), lambda: None): pass
        record = json.loads((self.out/'log-staging.json').read_text())
        self.assertFalse(record['logs_disabled'])
        self.assertFalse(record['published'])
        self.assertFalse(record['measurement_allowed'])
        self.assertTrue((self.stage.root/LOG_NAMES[0]).exists())

    def test_flush_failure_blocks_measurement_and_retains_source(self):
        def fail(): raise OSError('flush failed')
        with self.assertRaises(OSError):
            with StagedCapture(self.stage, self.logger(), fail): pass
        self.assertFalse((self.out/LOG_NAMES[0]).exists())
        self.assertTrue((self.stage.root/LOG_NAMES[0]).exists())


if __name__ == '__main__': unittest.main()

