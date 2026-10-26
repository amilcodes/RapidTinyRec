"""Exact FP32 serving environment and pinned runtime checks; Torch is lazy."""
FORBIDDEN_ENV = ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
                 'CUBLAS_WORKSPACE_CONFIG', 'CUBLASLT_WORKSPACE_SIZE',
                 'TORCH_CUBLASLT_UNIFIED_WORKSPACE', 'TORCH_BLAS_PREFER_CUBLASLT',
                 'TORCHINDUCTOR_MAX_AUTOTUNE', 'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM',
                 'TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE')


def validate_environment(env):
    if any(key in env for key in FORBIDDEN_ENV):
        raise ValueError('Uncontrolled serving precision/backend/workspace/autotune override')
    if any(env.get(key, '0') != '0' for key in ('PYTORCH_TUNABLEOP_ENABLED', 'PYTORCH_TUNABLEOP_TUNING')):
        raise ValueError('TunableOp must remain disabled')


def policy_state(backend):
    return dict(matmul_tf32=backend.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=backend.backends.cudnn.allow_tf32,
                intraop_threads=backend.get_num_threads(),
                float32_matmul_precision=backend.get_float32_matmul_precision(),
                blas_backend=str(backend.backends.cuda.preferred_blas_library()))


def validate_state(state):
    expected=dict(matmul_tf32=False,cudnn_tf32=False,intraop_threads=2,
                  float32_matmul_precision='highest',blas_backend='_BlasBackend.Cublas')
    if state != expected or state['matmul_tf32'] is not False or state['cudnn_tf32'] is not False or type(state['intraop_threads']) is not int:
        raise ValueError('Frozen exact FP32 serving policy changed')


def validate_runtime(backend, transformers_version):
    if (str(backend.__version__),backend.version.cuda,backend.version.git_version,transformers_version) != (
            '2.14.0+cu126','12.6','08187d9e0fba026dc8217405802ab5381dc88d90','4.55.4'):
        raise RuntimeError('Pinned serving runtime required')
    if not backend.cuda.is_available() or backend.cuda.device_count()!=1 or backend.cuda.get_device_name(0)!='NVIDIA RTX A6000':
        raise RuntimeError('One regular NVIDIA RTX A6000 required')
    state=policy_state(backend)
    if state['matmul_tf32'] is not False or state['float32_matmul_precision']!='highest' or state['blas_backend']!='_BlasBackend.Cublas':
        raise RuntimeError('Fresh process must start with highest FP32/default cuBLAS policy')
