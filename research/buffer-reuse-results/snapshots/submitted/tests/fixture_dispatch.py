import copy

import json

from pathlib import Path

import tempfile

import unittest

def report(root, wave=True, api_batch=None, launches=None, correlated=True):
    batch = (100 if wave else 50) if api_batch is None else api_batch
    n = (4 if wave else 8) if launches is None else launches
    (root / 'api.log').write_text(''.join((f'I! cuBLAS 12 function cublasSgemm(\nm: type=int; val={m}\nn: type=int; val={batch}\nk: type=int; val={k}\nMathMode=CUBLAS_TF32_TENSOR_OP_MATH\n' for m, k in ((8960, 1536), (1536, 8960)))))
    profiles = []
    for length, users in ((76, [1, 2]), (121, [6, 10])):
        events = [dict(name='cudaGraphLaunch', cat='cuda_runtime', args={'correlation': i}) for i in range(n)]
        events += [dict(name='ampere_s1688tf32_kernel', cat='kernel', dur=1, args={'correlation': i if correlated else 999}) for i in range(n)]
        name = f'{length}.json'
        (root / name).write_text(json.dumps({'traceEvents': events}))
        profiles.append(dict(prompt_tokens=length, cohort_indices=users, trace=name))
    return dict(capture_logs=dict(available=True, cublas='api.log'), profiles=profiles)

def counters(wave=True):

    def stage(n):
        prefix = dict(physical_prefix_batch=2 if wave else 1, suffix_batch=100 if wave else 50, max_prefix_tokens=121, max_suffix_tokens=3, requests_loaded=2 * n if wave else n)
        if wave:
            prefix['waves_loaded'] = n
        replay = dict(graphs=8, frozen=True, replays=4 * n, physical_prefix=prefix, events=[dict(cache_tokens=length + i, batch=100 if wave else 50) for length in (76, 121) for i in range(4)])
        reorder = dict(direct_calls=4 * n, fallback_calls=n, prefix_bytes_reordered=0)
        return (replay, reorder)
    setup = {}
    summary = dict(complete=True, measured_trials=12, measured_requests=12)
    dispatch = {}
    setup['replay'], setup['direct_reorder'] = stage(4 if wave else 8)
    summary['replay'], summary['direct_reorder'] = stage(10 if wave else 20)
    dispatch['replay_after_profiles'], dispatch['direct_reorder_after_profiles'] = stage(12 if wave else 24)
    return (setup, summary, dispatch)
