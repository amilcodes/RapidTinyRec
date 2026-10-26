import copy
from types import SimpleNamespace as NS
import unittest
from rapidtinyrec.precision_policy import precision_policy, validate_environment, validate_runtime, validate_state, TORCH_COMMIT


class FakeTorch:
    def __init__(self):
        self.backends=NS(cuda=NS(matmul=NS(allow_tf32=False),preferred_blas_library=lambda:'_BlasBackend.Cublas'),cudnn=NS(allow_tf32=True))
        self.__version__='2.14.0+cu126';self.version=NS(cuda='12.6',git_version=TORCH_COMMIT)
        self.cuda=NS(is_available=lambda:True,device_count=lambda:1,get_device_name=lambda i:'NVIDIA RTX A6000')
    def get_float32_matmul_precision(self):
        return 'high' if self.backends.cuda.matmul.allow_tf32 else 'highest'


class PrecisionPolicyTests(unittest.TestCase):
    def test_restore_on_failure_and_no_backend_mutation(self):
        torch=FakeTorch()
        with self.assertRaisesRegex(RuntimeError,'test'):
            with precision_policy(torch,'folded_tf32',{}):
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                raise RuntimeError('test')
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_drift_fails_and_restores(self):
        torch=FakeTorch()
        with self.assertRaisesRegex(RuntimeError,'Unexpected'):
            with precision_policy(torch,'folded_tf32',{}):torch.backends.cuda.matmul.allow_tf32=False
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)

    def test_no_silent_override_or_tuning(self):
        for env in ({'NVIDIA_TF32_OVERRIDE':'0'},{'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
                    {'TORCHINDUCTOR_MAX_AUTOTUNE':'0'},{'PYTORCH_TUNABLEOP_ENABLED':'1'}):
            with self.assertRaises(RuntimeError):validate_environment(env)
        validate_environment({'PYTORCH_TUNABLEOP_ENABLED':'0','PYTORCH_TUNABLEOP_TUNING':'0'})

    def test_runtime_rejects_wrong_wheel_or_unallocated_compute(self):
        t=FakeTorch();validate_runtime(t,'4.55.4',{'SLURM_JOB_ID':'123'})
        with self.assertRaises(RuntimeError):validate_runtime(t,'4.55.4',{})
        t.version.git_version='wrong'
        with self.assertRaises(RuntimeError):validate_runtime(t,'4.55.4',{'SLURM_JOB_ID':'123'})

    def test_requires_fresh_baseline_and_specific_variant(self):
        t=FakeTorch();t.backends.cuda.matmul.allow_tf32=True
        with self.assertRaises(RuntimeError):
            with precision_policy(t,'folded_tf32',{}):pass
        with self.assertRaises(ValueError):
            with precision_policy(FakeTorch(),'bf16',{}):pass
        with precision_policy(FakeTorch(),'folded_fp32',{}) as s:
            self.assertEqual(s['float32_matmul_precision'],'highest')

    def test_installed_legacy_setter_maps_to_high_without_dtype_change(self):
        try:import torch
        except ImportError:self.skipTest('Torch CPU check runs in decoding CI')
        old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            with precision_policy(torch,'folded_tf32',{}) as s:
                self.assertEqual(s['float32_matmul_precision'],'high')
                self.assertEqual(torch.ones(1,dtype=torch.float32).dtype,torch.float32)
        finally:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old
