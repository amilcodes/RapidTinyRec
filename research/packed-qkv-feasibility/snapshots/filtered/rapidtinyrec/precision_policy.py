"""One explicit CUDA matmul policy; no backend, workspace or algorithm sweep.

The legacy setter is deliberately shared with the accepted TF32-off baseline.
Pinned Context.cpp updates HIGH/HIGHEST and CUDA MATMUL TF32/IEEE together.
Do not mix this setter with the new hierarchical precision setters.
"""
from contextlib import contextmanager
import os

TORCH_COMMIT = '08187d9e0fba026dc8217405802ab5381dc88d90'
VARIANTS = ('folded_fp32', 'folded_tf32')
PHASES = (VARIANTS[0], VARIANTS[1], VARIANTS[1], VARIANTS[0])
# Reject even nominally benign overrides: this experiment has one frozen policy.
FORBIDDEN_ENV = ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
                 'CUBLAS_WORKSPACE_CONFIG', 'CUBLASLT_WORKSPACE_SIZE',
                 'TORCH_CUBLASLT_UNIFIED_WORKSPACE', 'TORCH_BLAS_PREFER_CUBLASLT',
                 'TORCHINDUCTOR_MAX_AUTOTUNE', 'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM',
                 'TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE')


def validate_environment(environ):
    present = {k: environ[k] for k in FORBIDDEN_ENV if k in environ}
    if present:
        raise RuntimeError(f'Uncontrolled precision/backend overrides: {present}')
    for key in ('PYTORCH_TUNABLEOP_ENABLED', 'PYTORCH_TUNABLEOP_TUNING'):
        if environ.get(key, '0') != '0':
            raise RuntimeError(f'{key} must be disabled')


def policy_state(torch):
    return dict(matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                blas_preference=str(torch.backends.cuda.preferred_blas_library()))


def validate_state(state, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    enabled = variant == VARIANTS[1]
    if state != dict(matmul_tf32=enabled, cudnn_tf32=False,
                     float32_matmul_precision='high' if enabled else 'highest',
                     blas_preference='_BlasBackend.Cublas'):
        raise RuntimeError(f'Unexpected numerical policy: {state}')


@contextmanager
def precision_policy(torch, variant, environ=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    validate_environment(os.environ if environ is None else environ)
    old = policy_state(torch)
    # A fresh process must begin with the accepted highest CUDA policy.
    if old['matmul_tf32'] or old['float32_matmul_precision'] != 'highest':
        raise RuntimeError(f'Expected fresh highest-precision process: {old}')
    if old['blas_preference'] != '_BlasBackend.Cublas':
        raise RuntimeError('Do not change BLAS backend in the precision experiment')
    try:
        torch.backends.cuda.matmul.allow_tf32 = variant == VARIANTS[1]
        torch.backends.cudnn.allow_tf32 = False
        state = policy_state(torch)
        validate_state(state, variant)
        yield state
        validate_state(policy_state(torch), variant)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = old['cudnn_tf32']


def validate_runtime(torch, transformers_version, environ=None):
    env = os.environ if environ is None else environ
    validate_environment(env)
    if not env.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated OSCAR compute only')
    if (str(torch.__version__), torch.version.cuda, torch.version.git_version,
            transformers_version) != ('2.14.0+cu126', '12.6', TORCH_COMMIT, '4.55.4'):
        raise RuntimeError('Pinned runtime/build identity mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one allocated CUDA device required')
    if torch.cuda.get_device_name(0) != 'NVIDIA RTX A6000':
        raise RuntimeError('Expected regular RTX A6000')
