"""BF16 storage policy with unchanged HF FP32 normalization and beam scoring."""
from contextlib import contextmanager
from .bf16_protocol import PHASES
from .precision_policy import precision_policy as tf32_policy, policy_state as base_state, validate_runtime

VARIANTS=('folded_tf32','folded_bf16')

def policy_state(torch):
    return dict(base_state(torch),
        bf16_reduced_precision_reduction=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        bf16_split_k=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k),
        math_sdpa_reduced_precision_reduction=bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()))

def validate_state(state,variant):
    assert variant in VARIANTS
    assert state==dict(matmul_tf32=True,cudnn_tf32=False,float32_matmul_precision='high',
        blas_preference='_BlasBackend.Cublas',bf16_reduced_precision_reduction=False,
        bf16_split_k=True,math_sdpa_reduced_precision_reduction=False)

@contextmanager
def precision_policy(torch,variant):
    assert variant in VARIANTS
    matmul=torch.backends.cuda.matmul
    old=(matmul.allow_bf16_reduced_precision_reduction,matmul.allow_bf16_reduced_precision_reduction_split_k)
    old_sdp=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    assert old[1] is True,'No split-K algorithm-policy sweep'
    try:
        matmul.allow_bf16_reduced_precision_reduction=(False,old[1])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        with tf32_policy(torch,'folded_tf32'):
            state=policy_state(torch);validate_state(state,variant)
            yield state
            validate_state(policy_state(torch),variant)
    finally:
        matmul.allow_bf16_reduced_precision_reduction=old
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(old_sdp)
