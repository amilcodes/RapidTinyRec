"""Independent native API and captured-kernel evidence reader (no torch import)."""
import collections
import json
import re
from pathlib import Path


def api_gemms(text):
    """Associate math mode with each GEMM block, not unrelated SetMathMode calls."""
    result = []
    for block in re.split(r'(?=I! cuBLAS .*? function )', text):
        if not re.search(r'function .*?cublasSgemm(?:_v2|StridedBatched|Batched)?\(', block):
            continue
        modes = re.findall(r'MathMode=(\w+)', block)
        shape = {}
        for key in ('m', 'n', 'k'):
            match = re.search(r'\b' + key + r': type=int; val=(\d+)', block)
            if match:
                shape[key] = int(match.group(1))
        result.append(dict(math_mode=modes[-1] if modes else None, shape=shape))
    return result


def trace_kernels(trace):
    events = trace.get('traceEvents', [])
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('dur', 0) > 0]
    counts = collections.Counter(e.get('name', '') for e in kernels)
    launches = {e['args']['correlation'] for e in events
                if e.get('name') == 'cudaGraphLaunch' and e.get('cat') in ('cuda_runtime','cuda_driver')
                and 'correlation' in e.get('args', {})}
    replay_kernels = [e for e in kernels if e.get('args', {}).get('correlation') in launches]
    replay_counts = collections.Counter(e.get('name', '') for e in replay_kernels)
    # Corroboration must be linked to the frozen decode graph, not prefill.
    tf32 = {k: v for k, v in replay_counts.items() if re.search(r'(^|[_:])(?:tf32|s1688tf32|s1688gemm)(?:[_:]|$)', k, re.I)}
    return dict(kernel_events=len(kernels), kernels=dict(counts), tf32_kernel_symbols=tf32,
                graph_kernel_events=len(replay_kernels),graph_kernels=dict(replay_counts),
                cuda_graph_launches=len(launches))


def dispatch_evidence(root, report, candidate):
    root = Path(root)
    logs = report['capture_logs']
    calls = api_gemms((root / logs['cublas']).read_text()) if logs.get('available') else []
    modes = collections.Counter(c['math_mode'] for c in calls)
    traces = [trace_kernels(json.loads((root / t['trace']).read_text())) for t in report['profiles']]
    desired = 'CUBLAS_TF32_TENSOR_OP_MATH' if candidate else 'CUBLAS_DEFAULT_MATH'
    # Decode native MLP/head shape evidence: not an unrelated small TF32 probe.
    required_shapes = [(8960, 50, 1536), (1536, 50, 8960)]
    covered = {str(shape): sum(c['shape'] == dict(zip(('m', 'n', 'k'), shape)) and c['math_mode'] == desired
                               for c in calls) for shape in required_shapes}
    native_policy = bool(calls) and all(covered.values()) and modes[desired] > 0
    if not candidate:native_policy = native_policy and set(modes)=={desired}
    replay = len(traces) == 2 and all(t['graph_kernel_events'] > 0 and t['cuda_graph_launches'] >= 4 for t in traces)
    accelerated = all(t['tf32_kernel_symbols'] for t in traces) if candidate else not any(t['tf32_kernel_symbols'] for t in traces)
    return dict(verified=bool(logs.get('available') and native_policy and replay and accelerated),
                api_gemm_calls=len(calls), api_math_modes=dict(modes), required_shape_calls=covered,
                replay_profiles=traces, native_policy_verified=native_policy,
                replay_verified=replay, kernel_symbol_corroboration=accelerated,
                limitation='Native GEMM API math mode plus TF32 kernel symbols in frozen model replay corroborate dispatch. Symbols alone are not proof; not every GEMM is required to accelerate. No instruction-level disassembly or speed claim.')
