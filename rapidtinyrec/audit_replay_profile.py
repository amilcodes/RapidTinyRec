"""Audit post-benchmark profiles and attribute actual CUDA activities to scopes.

Run audit_decode_pilot first for provenance and uninstrumented measurements.
This audit is stdlib-only. Instrumented wall times are not latency benchmarks.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import statistics

from .audit_decode_pilot import compare, digest, read, validate_run


def union_us(events):
    total = 0.
    end = -math.inf
    for event in sorted(events, key=lambda e: e['ts']):
        stop = event['ts'] + event['dur']
        total += max(0., stop - max(end, event['ts']))
        end = max(end, stop)
    return total


def trace_summary(path):
    with gzip.open(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e['name'].startswith('section/')]
    cpus = {e.get('args', {}).get('External id'): e for e in events if e.get('cat') == 'cpu_op'}
    launches = {e['args']['correlation']: e for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and e.get('ph') == 'X'
                and 'correlation' in e.get('args', {})}
    activities = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    stages = collections.defaultdict(lambda: {'kernel_ms': 0., 'copy_ms': 0., 'kernels': 0, 'copies': 0})
    for event in activities:
        args = event.get('args', {})
        anchor = launches.get(args.get('correlation')) or cpus.get(args.get('External id'))
        parents = [s for s in scopes if anchor and s['tid'] == anchor['tid']
                   and s['ts'] <= anchor['ts'] < s['ts'] + s['dur']]
        stage = min(parents, key=lambda s: s['dur'])['name'][8:] if parents else 'unassigned'
        if stage == 'request':
            stage = 'generation_other'
        kernel = event['cat'] == 'kernel'
        stages[stage]['kernel_ms' if kernel else 'copy_ms'] += event['dur'] / 1000
        stages[stage]['kernels' if kernel else 'copies'] += 1
    assert sum(s['kernels'] + s['copies'] for s in stages.values()) == len(activities)
    kernels = [e for e in activities if e['cat'] == 'kernel']
    runtime = collections.Counter(e['name'] for e in launches.values())
    return {'sha256': digest(path), 'activity_count': len(activities),
            'kernel_count': len(kernels), 'kernel_sum_ms': sum(e['dur'] for e in kernels) / 1000,
            'gpu_activity_union_ms': union_us(activities) / 1000,
            'by_stage': dict(stages), 'runtime_calls': dict(runtime)}


def audit(root, snapshot):
    profile = read(root / 'replay_profile.json')
    summary = read(root / 'summary.json')
    manifest = read(root / 'manifest.json')
    requests = read(root / 'requests.json')
    assert profile['complete'] and profile['measurement_compilations'] == 0
    assert manifest['profile_after_measurement']
    for name, value in profile['source_sha256'].items():
        assert digest(snapshot / name) == value == manifest['sha256'][name]
    assert profile['before_replay'] == summary['replay']
    before, after = profile['before_replay'], profile['after_replay']
    assert before['events'] == after['events'] and before['graphs'] == after['graphs'] == 8
    assert after['replays'] - before['replays'] == 32
    assert len(profile['light_profiles']) == 12 and len(profile['torch_profiles']) == 4
    refs = {(r['cohort_index'], v): run for r in requests for v, run in r['runs'].items()}
    representatives = {length: next(u['cohort_index'] for u in manifest['users'] if u['prompt_tokens'] == length)
                       for length in (76, 121)}
    seen, pairs, light = set(), [], collections.defaultdict(list)
    traces = []
    for kind, rows in [('light', profile['light_profiles']), ('torch', profile['torch_profiles'])]:
        for row in rows:
            length, variant = row['prompt_tokens'], row['variant']
            assert row['cohort_index'] == representatives[length]
            assert variant in ('compiled_gqa', 'replay')
            identity = (kind, length, variant, row.get('repeat', 0))
            assert identity not in seen
            seen.add(identity)
            if kind == 'light':
                assert row['repeat'] in (0, 1, 2)
            ref = refs[(row['cohort_index'], variant)]
            validate_run({**row['output'], 'elapsed_s': 1.})
            pairs.append((ref, row['output']))
            if kind == 'light':
                scopes = row['scopes']
                request = next(s for s in scopes if s['name'] == 'request')
                children = [s for s in scopes if s['parent'] == 'request']
                assert math.isclose(sum(s['wall_ms'] for s in children) + request['exclusive_wall_ms'], request['wall_ms'], abs_tol=1e-6)
                counts = collections.Counter(s['name'] for s in scopes)
                assert counts['compiled_prefill'] == 1 and counts['cache_reorder'] == 5
                if variant == 'replay':
                    assert counts['replay_input_staging'] == counts['graph_replay'] == 4
                    assert counts['compiled_decode'] == 0
                else:
                    assert counts['compiled_decode'] == 4 and counts['graph_replay'] == 0
                host = collections.defaultdict(float)
                for scope in scopes:
                    host[scope['name']] += scope['exclusive_wall_ms']
                assert math.isclose(sum(host.values()), request['wall_ms'], abs_tol=1e-6)
                light[(length, variant)].append(dict(host))
            else:
                path = root / row['trace']
                assert path.parent == root and path.name == f'replay_trace_{variant}_{length}.json.gz'
                traces.append({'prompt_tokens': length, 'variant': variant, **trace_summary(path)})
    comparison = compare(pairs)
    assert comparison['matching_rank_lists'] == comparison['matching_sequence_vectors'] == comparison['matching_transition_arrays'] == 16
    return {'integrity_audit_passed': True, 'instrumented_output_comparison': comparison,
            'light_exclusive_host_ms_mean': [
                {'prompt_tokens': length, 'variant': variant, 'repeats': len(rows),
                 'stages': {k: statistics.mean(r.get(k, 0.) for r in rows) for k in rows[0]}}
                for (length, variant), rows in light.items()],
            'traces': traces, 'measurement_compilations': 0, 'captures_during_profiling': 0,
            'audit_source_sha256': digest(Path(__file__)),
            'note': 'Host call times overlap asynchronous GPU execution. Do not add host and GPU columns. CUDA event spans include idle and are not used as GPU compute. Trace activity durations include profiling effects; use the separate uninstrumented audit for latency. One trace per variant and length.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results, args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
