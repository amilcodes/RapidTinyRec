"""Fixed diagnostic scope and backend lifecycle; importing this module needs no GPU."""
from contextlib import contextmanager

BACKENDS = ('default', 'cublas', 'cublaslt')
USERS = (1, 2, 6, 10)
LAYERS = (0, 13, 27)
ROLES = ('gate', 'up', 'down')
TORCH_COMMIT = '08187d9e0fba026dc8217405802ab5381dc88d90'


@contextmanager
def backend_scope(api, backend):
    if backend not in BACKENDS:
        raise ValueError(backend)
    previous = api()
    try:
        api(backend)
        yield str(api())
    finally:
        api(previous)


def expected_shapes(role):
    if role not in ROLES:
        raise ValueError(role)
    inner, outer = (8960, 1536) if role == 'down' else (1536, 8960)
    return (50, inner), (outer, inner), (50, outer)


def validate_tensors(x, weight, output, role):
    xs, ws, ys = expected_shapes(role)
    for tensor, shape in ((x, xs), (weight, ws), (output, ys)):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != (shape[1], 1):
            raise ValueError(f'{role}: wrong shape/stride')
        if str(tensor.dtype) != 'torch.float32':
            raise ValueError('FP32 required')
    pointers = [t.untyped_storage().data_ptr() for t in (x, weight, output)]
    if len(set(pointers)) != 3:
        raise ValueError('projection input/weight/output storage aliases')


def case_keys():
    return [(u, layer) for u in USERS for layer in LAYERS]
