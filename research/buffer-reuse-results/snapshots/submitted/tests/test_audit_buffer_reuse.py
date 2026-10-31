"""Synthetic identity and failure-closure boundaries; no release data/bootstrap."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import test_buffer_reuse_allocation as allocation_fixture
from rapidtinyrec.audit_buffer_reuse import validate_header, validate_work_identity, audit, inspect_phases, evaluate_evidence, TORCH_COMMIT
from rapidtinyrec.buffer_reuse_protocol import PROTOCOL, PROTOCOL_SHA256, load_protocol, PHASES
from rapidtinyrec.audit_buffer_reuse_outputs import validate_request_group
from fixture_outputs import request
ROOT = Path(__file__).resolve().parents[1]

class RawAuditBoundaryTests(unittest.TestCase):

    def setUp(self):
        fixture = allocation_fixture.AllocationIdentityTests()
        fixture.setUp()
        self.live = fixture.records()[0]
        self.protocol = load_protocol(ROOT / PROTOCOL)
        self.users = [dict(cohort_index=i) for i in (1, 2, 6, 10)]
        self.expected = dict(users_sha256='users', release_sha256={'config': 'hash'})
        self.reference = dict(manifest_sha256='reference')
        self.header = dict(schema='released-buffer-reuse-phase-v1', experiment=self.protocol['experiment'], phase=1, half=0, variant=PHASES[1], phase_protocol=self.protocol['phases'][1], protocol_sha256=PROTOCOL_SHA256, job_id=self.live['job_id'], run_id=self.live['run_id'], allocation=deepcopy(self.live), users=self.users, cohort_sha256='users', reference_manifest_sha256='reference', dtype='float32', tf32=True, attention_layout='folded_query_heads', hardware='NVIDIA RTX A6000', torch_git=TORCH_COMMIT, cuda_version='12.6', versions={'torch': '2.14.0+cu126', 'transformers': '4.55.4', 'tokenizers': '0.21.4'}, compiler_policy=dict(max_autotune=False, max_autotune_gemm=False, max_autotune_pointwise=False), api_log_staging=True, timing_contract=self.protocol['timing'], precision_policy=dict(matmul_tf32=True, cudnn_tf32=False, float32_matmul_precision='high', blas_preference='_BlasBackend.Cublas'), release_verification=dict(schema='precision-release-verification-v1', observed_sha256={'config': 'hash'}, elapsed_s=0.01, boundary='before_tokenizer_and_model_load'))
        self.work = dict(schema='released-buffer-reuse-work-v1', job_id=self.live['job_id'], run_id=self.live['run_id'], protocol_sha256=PROTOCOL_SHA256, started_monotonic=100.0, deadline_monotonic=1720.0)

    def validate(self):
        return validate_header(self.header, 1, self.live, self.protocol, self.users, self.expected, self.reference)

    def test_native_header_accepts_and_rejects_relabeling(self):
        self.validate()
        for key, value in [('schema', 'precision-paired-staged-phase-v1'), ('variant', 'folded_tf32'), ('phase', 2), ('job_id', 'other'), ('protocol_sha256', 'other'), ('api_log_staging', False), ('dtype', 'bfloat16')]:
            self.setUp()
            self.header[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_original_allocation_proof_cannot_be_replaced(self):
        for key, value in [('scheduler_command', ['true']), ('requires_local_oscar_reconciliation', False), ('uploaded_manifest_sha256', 'different'), ('scheduler_stdout', 'JobId=12345')]:
            self.setUp()
            self.header['allocation']['allocation_identity'][key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, AssertionError, KeyError)):
                self.validate()
        self.setUp()
        self.header['allocation']['submission'] = 'submitted'
        with self.assertRaises(ValueError):
            self.validate()

    def test_source_hash_and_resource_binding_required(self):
        for key in ('sha256', 'resources', 'files', 'remote_dir'):
            self.setUp()
            self.header['allocation'][key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_work_identity_and_inclusive_deadline_are_fixed(self):
        validate_work_identity(self.work, self.live)
        for key, value in [('job_id', 'other'), ('run_id', 'other'), ('protocol_sha256', 'other'), ('deadline_monotonic', 1721.0), ('deadline_monotonic', float('inf')), ('started_monotonic', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_work_identity(dict(self.work, **{key: value}), self.live)

    def test_all_entrypoints_reject_local_before_dependencies_or_data(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Allocated'):
                audit('/missing', '/missing', '/missing', '/missing')
            with self.assertRaisesRegex(ValueError, 'Allocated'):
                inspect_phases(None, None, None, None, self.live)
            with self.assertRaisesRegex(ValueError, 'Allocated'):
                evaluate_evidence(None, None, None, self.live)

    def test_cli_preserves_failure_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'audit.json'
            env = {k: v for k, v in os.environ.items() if k != 'SLURM_JOB_ID'}
            command = [sys.executable, '-m', 'rapidtinyrec.audit_buffer_reuse']
            for name in ('root', 'code', 'historical', 'release'):
                command += ['--' + name, '/missing']
            command += ['--output', str(output)]
            first = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 1)
            before = output.read_bytes()
            failure = json.loads(before)
            self.assertFalse(failure['integrity_audit_passed'])
            self.assertIn('Allocated buffer-reuse', failure['error'])
            second = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 1)
            self.assertEqual(output.read_bytes(), before)
            self.assertIn('Preserve existing audit evidence', second.stderr)
if __name__ == '__main__':
    unittest.main()
