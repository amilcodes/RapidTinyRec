"""Audit replay pilot provenance, FP32 bits and latency without importing torch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

VARIANTS = ('compiled_gqa', 'replay')
SOURCE_FILES = {'pilot_decode_replay.py', 'decode_replay.py', 'compiled_forward.py',
                'shared_prefill.py', 'gpu_constraints.py', 'expected.json'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bits(values):
    return [struct.pack('!f', value) for value in values]


def flat(run, field):
    values = run[field]
    return values if field == 'sequence_logp' else [x for row in values for x in row]


def validate_run(run):
    assert len(run['token_ids']) == len(run['sequence_logp']) == len(run['transition_logp']) == 50
    assert len({tuple(t) for t in run['token_ids']}) == 50
    assert all(len(t) == 5 and t[-2:] == [198, 151645] for t in run['token_ids'])
    assert all(len(t) == 5 for t in run['transition_logp'])
    assert math.isfinite(run['elapsed_s']) and run['elapsed_s'] > 0
    assert all(math.isfinite(x) for f in ('sequence_logp', 'transition_logp') for x in flat(run, f))
    for score, transitions in zip(run['sequence_logp'], run['transition_logp']):
        total = 0.
        for transition in transitions:
            total = struct.unpack('!f', struct.pack('!f', total + transition))[0]
        assert bits([total]) == bits([score]), 'Five-token FP32 sum differs'


def compare(pairs):
    report = {'requests': len(pairs), 'rank_positions': 50 * len(pairs),
              'sequence_values': 50 * len(pairs), 'transition_values': 250 * len(pairs),
              'matching_rank_lists': 0, 'matching_rank_positions': 0,
              'matching_sequence_vectors': 0, 'matching_transition_arrays': 0,
              'matching_sequence_values': 0, 'matching_transition_values': 0,
              'max_sequence_difference_common': 0., 'max_transition_difference_common': 0.}
    for left, right in pairs:
        report['matching_rank_lists'] += left['token_ids'] == right['token_ids']
        report['matching_rank_positions'] += sum(a == b for a, b in zip(left['token_ids'], right['token_ids']))
        for field, label, vector in [('sequence_logp', 'sequence', 'vectors'),
                                     ('transition_logp', 'transition', 'arrays')]:
            a, b = (bits(flat(run, field)) for run in (left, right))
            report['matching_' + label + '_' + vector] += a == b
            report['matching_' + label + '_values'] += sum(x == y for x, y in zip(a, b))
            # Deltas follow the same candidate if ranks change; equality counts
            # above deliberately compare rank positions, never a sorted set.
            x, y = ({tuple(t): s for t, s in zip(run['token_ids'], run[field])} for run in (left, right))
            for token in x.keys() & y.keys():
                a, b = ([x[token]], [y[token]]) if label == 'sequence' else (x[token], y[token])
                key = 'max_' + label + '_difference_common'
                report[key] = max(report[key], max(abs(i - j) for i, j in zip(a, b)))
    return report


def latency(rows):
    result = {}
    for variant in VARIANTS:
        values = [r['runs'][variant]['elapsed_s'] * 1000 for r in rows]
        result[variant] = {'n': len(values), 'median_ms': statistics.median(values),
                           'p95_ms': statistics.quantiles(values, n=100, method='inclusive')[94]}
    return result


def audit(root, historical, snapshot, run_record):
    manifest, summary, setup = (read(root / name) for name in ('manifest.json', 'summary.json', 'setup.json'))
    rows, compiler = read(root / 'requests.json'), read(root / 'compilation.json')
    reference, expected = read(historical / 'manifest.json'), read(snapshot / 'expected.json')
    assert expected == read(Path(__file__).with_name('expected.json')), 'Pinned release/cohort changed'
    assert manifest['experiment'] == 'decode_replay_pilot' and summary['complete']
    assert manifest['dtype'] == 'float32' and manifest['tf32'] is False
    assert manifest['hardware'] == 'NVIDIA RTX A6000'
    assert manifest['versions'] == {'torch': '2.14.0+cu126', 'transformers': '4.55.4', 'tokenizers': '0.21.4'}
    assert reference['experiment'] == 'gqa_three_way' and reference['dtype'] == 'float32'
    assert json_digest({k: v for k, v in reference.items() if k != 'manifest_sha256'}) == reference['manifest_sha256']
    assert manifest['reference_manifest_sha256'] == reference['manifest_sha256']
    assert len(reference['users']) == 500
    assert json_digest(reference['users']) == expected['users_sha256'] == manifest['cohort_sha256']
    lengths = sorted({u['prompt_tokens'] for u in reference['users']})
    selected = [u for length in (lengths[0], lengths[-1])
                for u in [u for u in reference['users'] if u['prompt_tokens'] == length][:2]]
    assert len(selected) == 4 and manifest['users'] == selected
    source_files = SOURCE_FILES | ({'profile_decode_replay.py', 'profile_shared.py'}
                                  if manifest.get('profile_after_measurement') else set())
    assert set(manifest['sha256']) == source_files
    for name, value in manifest['sha256'].items():
        assert digest(snapshot / name) == value, name
    record = read(run_record)
    assert record['resources']['gpus'] == 1 and record['resources']['gpu_type'] == 'nvidia_rtx_a6000'
    for name, value in record['sha256'].items():
        assert digest(snapshot.parent / name) == value, name
    assert len(record['sha256']) == len(record['files'])
    assert compiler['phase'] == 'complete' and compiler['measurement_compilations'] == 0
    assert compiler['events'] and all(e.get('success') or e.get('expected_restart') for e in compiler['events'])
    assert not compiler['compiler_counters'].get('graph_break')
    assert not compiler['compiler_counters'].get('unimplemented')
    before, after = setup['replay'], summary['replay']
    assert before['frozen'] is True and after['frozen'] is True
    assert before['graphs'] == after['graphs'] == 8
    assert before['events'] == after['events'] and len(after['events']) == 8
    assert sorted(e['cache_tokens'] for e in after['events']) == sorted(n + s for n in (lengths[0], lengths[-1]) for s in range(4))
    assert all(e['batch'] == 50 and e['capture_and_warmup_s'] > 0 and e['input_buffer_bytes'] > 0 for e in after['events'])
    assert before['input_buffer_bytes'] == after['input_buffer_bytes'] == sum(e['input_buffer_bytes'] for e in after['events'])
    assert before['capture_and_warmup_s'] == after['capture_and_warmup_s'] == sum(e['capture_and_warmup_s'] for e in after['events'])
    assert setup['combined_setup_and_warmup_s'] >= after['capture_and_warmup_s']
    assert summary['reserved_bytes'] >= summary['allocated_bytes'] >= after['input_buffer_bytes']
    assert len(rows) == summary['measured_pairs'] and len(rows) >= 8 and len(rows) % 8 == 0
    repeats = len(rows) // 4
    assert after['replays'] - before['replays'] == 4 * len(rows)
    assert [(r['repeat'], r['cohort_index']) for r in rows] == [(i, u['cohort_index']) for i in range(repeats) for u in selected]
    seen, control_pairs = {}, []
    for row in rows:
        user = next(u for u in selected if u['cohort_index'] == row['cohort_index'])
        assert row['prompt_tokens'] == user['prompt_tokens']
        assert row['order'] == list(VARIANTS if row['repeat'] % 2 == 0 else reversed(VARIANTS))
        assert set(row['runs']) == set(VARIANTS)
        for variant, run in row['runs'].items():
            validate_run(run)
            key = (row['cohort_index'], variant)
            if key in seen:
                previous = seen[key]
                assert previous['token_ids'] == run['token_ids'], 'Repeated ranking drift'
                assert all(bits(flat(previous, f)) == bits(flat(run, f)) for f in ('sequence_logp', 'transition_logp')), 'Repeated score drift'
            seen[key] = run
        old = read(historical / 'users' / f"{row['cohort_index']:04d}.json")
        assert old['cohort_index'] == row['cohort_index'] and old['manifest_sha256'] == reference['manifest_sha256']
        control_pairs.append((old['runs']['compiled_gqa'], row['runs']['compiled_gqa']))
    comparison = compare([(r['runs']['compiled_gqa'], r['runs']['replay']) for r in rows])
    for key, field in [('matching_rank_lists', 'token_ids'), ('matching_sequence_vectors', 'sequence_logp'),
                       ('matching_transition_arrays', 'transition_logp')]:
        # The producer uses numeric equality, which equates +0 and -0. Validate
        # that summary independently; the report above requires exact FP32 bits.
        assert sum(r['runs']['compiled_gqa'][field] == r['runs']['replay'][field] for r in rows) == summary[key]
    timing = latency(rows)
    grouped = {str(length): latency([r for r in rows if r['prompt_tokens'] == length]) for length in (lengths[0], lengths[-1])}
    for actual, stored in [(timing, summary['latency'])] + [(grouped[n], summary['latency_by_length'][n]) for n in grouped]:
        for variant in VARIANTS:
            assert actual[variant]['n'] == stored[variant]['n']
            for stat in ('median', 'p95'):
                assert math.isclose(actual[variant][stat + '_ms'], stored[variant][stat + '_s'] * 1000, rel_tol=1e-12)
    return {'integrity_audit_passed': True,
            'scope': 'Four unique users repeated; not a 500-user latency or correctness estimate.',
            'unique_users': 4, 'repeats': repeats, 'cohort_sha256': manifest['cohort_sha256'],
            'comparison': comparison, 'historical_control': compare(control_pairs),
            'latency_ms': timing, 'latency_by_length_ms': grouped,
            'median_reduction_percent': 100 * (1 - timing['replay']['median_ms'] / timing['compiled_gqa']['median_ms']),
            'p95_reduction_percent': 100 * (1 - timing['replay']['p95_ms'] / timing['compiled_gqa']['p95_ms']),
            'combined_setup_and_warmup_s': setup['combined_setup_and_warmup_s'],
            'backend_compile_total_s': compiler['backend_compile_total_s'],
            'replay': after, 'allocated_bytes': summary['allocated_bytes'], 'reserved_bytes': summary['reserved_bytes'],
            'measurement_compilations': 0, 'snapshot_files_verified': len(record['files']),
            'audit_script_sha256': digest(Path(__file__)),
            'note': 'Integrity passing does not require control/replay equality. Setup, compile and capture timings overlap; do not add them.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True, help='Frozen rapidtinyrec package directory')
    parser.add_argument('--run-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.results, args.historical, args.snapshot, args.run_record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
