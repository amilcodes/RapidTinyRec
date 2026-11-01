"""Explicit, process-scoped precision for the sequential folded runner.

BF16 is an experimental quality tradeoff. Select before model creation; never
change dtype/policy on a live compiled callable or captured graph pool.
"""
from contextlib import contextmanager
import os

PRECISIONS = ('fp32', 'bf16-experimental')
TORCH_COMMIT = '08187d9e0fba026dc8217405802ab5381dc88d90'
FORBIDDEN_ENV = ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
                 'CUBLAS_WORKSPACE_CONFIG', 'CUBLASLT_WORKSPACE_SIZE',
                 'TORCH_CUBLASLT_UNIFIED_WORKSPACE', 'TORCH_BLAS_PREFER_CUBLASLT',
                 'TORCHINDUCTOR_MAX_AUTOTUNE', 'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM',
                 'TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE')


def model_dtype(torch, precision):
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown precision: {precision}')
    return torch.float32 if precision == 'fp32' else torch.bfloat16


def policy_state(torch):
    mm = torch.backends.cuda.matmul
    return dict(matmul_tf32=bool(mm.allow_tf32),
                cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                blas_preference=str(torch.backends.cuda.preferred_blas_library()),
                bf16_reduced_precision_reduction=bool(mm.allow_bf16_reduced_precision_reduction),
                bf16_split_k=bool(mm.allow_bf16_reduced_precision_reduction_split_k),
                math_sdpa_reduced_precision_reduction=bool(
                    torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))


def validate_runtime(torch, transformers_version):
    if (str(torch.__version__), torch.version.cuda, torch.version.git_version,
            transformers_version) != ('2.14.0+cu126', '12.6', TORCH_COMMIT, '4.55.4'):
        raise RuntimeError('This experimental runner requires the validated Torch/CUDA/Transformers build')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One visible CUDA device required')
    if torch.cuda.get_device_name(0) != 'NVIDIA RTX A6000':
        raise RuntimeError('This release is validated on NVIDIA RTX A6000 only')


@contextmanager
def precision_policy(torch, precision='fp32', environ=None):
    model_dtype(torch, precision)  # Reject before changing global state.
    env = os.environ if environ is None else environ
    overrides = [k for k in FORBIDDEN_ENV if k in env]
    overrides += [k for k in ('PYTORCH_TUNABLEOP_ENABLED', 'PYTORCH_TUNABLEOP_TUNING')
                  if env.get(k, '0') != '0']
    if overrides:
        raise RuntimeError(f'Uncontrolled precision/backend overrides: {overrides}')
    old = policy_state(torch)
    if old['blas_preference'] != '_BlasBackend.Cublas':
        raise RuntimeError('Expected the validated cuBLAS backend')
    if precision == 'bf16-experimental' and not old['bf16_split_k']:
        raise RuntimeError('Expected the reviewed BF16 split-K setting')
    mm = torch.backends.cuda.matmul
    try:
        mm.allow_tf32 = precision == 'bf16-experimental'
        torch.backends.cudnn.allow_tf32 = False
        if precision == 'bf16-experimental':
            mm.allow_bf16_reduced_precision_reduction = (False, old['bf16_split_k'])
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        active = policy_state(torch)
        expected = dict(old, matmul_tf32=precision == 'bf16-experimental',
                        cudnn_tf32=False,
                        float32_matmul_precision='highest' if precision == 'fp32' else 'high')
        if precision == 'bf16-experimental':
            expected.update(bf16_reduced_precision_reduction=False,
                            math_sdpa_reduced_precision_reduction=False)
        if active != expected:
            raise RuntimeError(f'Precision policy did not take effect: {active}')
        yield active
        if policy_state(torch) != active:
            raise RuntimeError('Precision policy changed while graphs were live')
    finally:
        # Restore the exact original high/highest/medium setting, not just TF32.
        torch.set_float32_matmul_precision(old['float32_matmul_precision'])
        torch.backends.cudnn.allow_tf32 = old['cudnn_tf32']
        mm.allow_bf16_reduced_precision_reduction = (
            old['bf16_reduced_precision_reduction'], old['bf16_split_k'])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(
            old['math_sdpa_reduced_precision_reduction'])


def validate_output(torch, output, precision, graph_logit_pointers=()):
    """HF owns FP32 score outputs; graph logits must never escape as scores."""
    if len(output.scores) != 5 or output.sequences_scores.dtype != torch.float32:
        raise RuntimeError('Expected all five FP32 scores and FP32 sequence scores')
    if any(s.dtype != torch.float32 for s in output.scores):
        raise RuntimeError('Normalization and beam scores must remain FP32')
    pointers = [s.untyped_storage().data_ptr() for s in output.scores]
    if len(set(pointers)) != 5 or set(pointers) & set(graph_logit_pointers):
        raise RuntimeError('Saved scores alias each other or borrowed graph logits')
    model_dtype(torch, precision)
