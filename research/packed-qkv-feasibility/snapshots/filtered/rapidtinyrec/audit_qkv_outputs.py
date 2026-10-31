"""Allocated QKV output audit components; no model or generation imports.

The future allocated orchestrator must authenticate all files and provenance.
These checks do not authorize submission or certify dispatch/allocation/quality.
Numerical failures are returned with their comparisons, never silently dropped.
"""
import math
import re
import statistics
import struct

ARMS = ('original_bf16', 'packed_qkv_bf16', 'packed_qkv_bf16', 'original_bf16')
USERS = {1: 76, 2: 76, 6: 121, 10: 121}
FIELDS = ('token_ids', 'beam_indices', 'sequence_logp', 'transition_logp', 'full_score_hashes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    require(type(value) in (float, int) and math.isfinite(value), 'Nonfinite or nonnumeric selected score')
    try:
        raw = struct.pack('!f', value)
    except (OverflowError, struct.error) as error:
        raise ValueError('Selected score outside FP32') from error
    require(struct.unpack('!f', raw)[0] == value, 'Selected score is not exactly representable as FP32')
    return raw


def owners(values):
    require(isinstance(values, list) and all(type(p) is int and p > 0 for p in values)
            and len(values) == len(set(values)), 'Invalid or duplicate storage owners')
    return set(values)


def validate_output(run, user, vocab_size, *, required_supplied=(), required_graph_logits=()):
    require(run['cohort_index'] == user['cohort_index'] and run['prompt_tokens'] == user['prompt_tokens'], 'Output request identity differs')
    for name in ('token_ids', 'beam_indices', 'transition_logp'):
        require(isinstance(run[name], list) and len(run[name]) == 50
                and all(isinstance(row, list) and len(row) == 5 for row in run[name]), 'Native beam50/five-step output shape differs: ' + name)
    require(all(type(x) is int and 0 <= x < vocab_size for row in run['token_ids'] for x in row), 'Token outside vocabulary')
    require(all(type(x) is int and -1 <= x < 50 for row in run['beam_indices'] for x in row), 'Ancestry outside request-local beam50')
    require(isinstance(run['sequence_logp'], list) and len(run['sequence_logp']) == 50, 'Missing sequence scores')
    sums = []
    for score, transitions in zip(run['sequence_logp'], run['transition_logp']):
        total = 0.0
        for value in transitions:
            f32(value)
            total = struct.unpack('!f', struct.pack('!f', total + value))[0]
        sums.append(f32(score) == f32(total))
    require(run['saved_score_dtypes'] == ['torch.float32'] * 5, 'Changed full score dtype')
    hashes = run['full_score_hashes']
    require(isinstance(hashes, list) and len(hashes) == 5, 'Missing full score hashes')
    for row in hashes:
        require(set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == [50, vocab_size]
                and row['dtype'] == 'torch.float32' and isinstance(row['sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Full score hash contract differs')
    saved = owners(run['saved_score_storage_ptrs'])
    supplied = owners(run['supplied_storage_ptrs'])
    logits = owners(run['graph_logit_storage_ptrs'])
    require(len(saved) == 5 and not saved & (supplied | logits | set(required_supplied) | set(required_graph_logits)), 'Score ownership aliases supplied tensors')
    require(set(required_supplied) <= supplied and set(required_graph_logits) <= logits, 'Omitted supplied or graph owners')
    if 'elapsed_s' in run:
        require(type(run['elapsed_s']) in (int, float) and math.isfinite(run['elapsed_s']) and run['elapsed_s'] > 0, 'Invalid measured request duration')
    return dict(five_step_fp32_sum_exact=all(sums), matching_sequence_sums=sum(sums), score_ownership_verified=True)


def compare_outputs(left, right, *, full_scores=True):
    """Bitwise selected-score comparison includes signed zero; do not sort ranks."""
    fields = FIELDS if full_scores else FIELDS[:-1]
    result = {}
    for field in fields:
        a, b = left[field], right[field]
        if field == 'sequence_logp':
            a, b = [f32(x) for x in a], [f32(x) for x in b]
        elif field == 'transition_logp':
            a, b = [[f32(x) for x in row] for row in a], [[f32(x) for x in row] for row in b]
        result[field] = a == b
    return dict(exact=all(result.values()), fields=result, full_score_comparison=full_scores)


def sequence_validity(run, constraints, root_prefix):
    valid = []
    for tokens in run['token_ids']:
        valid.append(all(token in constraints.get(tuple(root_prefix) if step == 0 else tuple(tokens[:step]), ())
                         for step, token in enumerate(tokens)))
    distinct = len({tuple(row) for row in run['token_ids']})
    return dict(valid_rows=sum(valid), rows=50, distinct_rows=distinct, passed=all(valid) and distinct == 50)


def audit_phase_outputs(phase, requests, warmups, profiles, users, vocab_size, constraints, root_prefix,
                        *, required_immutable=(), required_supplied=(), required_graph_logits=()):
    require(type(phase) is int and phase in range(4), 'Unknown phase')
    require({u['cohort_index']: u['prompt_tokens'] for u in users} == USERS and len(users) == 4, 'Declared cohort differs')
    by_user = {u['cohort_index']: u for u in users}
    half = phase // 2
    # Independently spell the schedule; never import the producer's schedule.
    expected = [(repeat, index) for repeat in range(half * 3, half * 3 + 3)
                for index in ([2, 1, 10, 6] if repeat % 2 else [1, 2, 6, 10])]
    require([(r['repeat'], r['cohort_index']) for r in requests] == expected, 'Timed request partition/order differs')
    expected_warmups = {f'{i}_{mode}' for i in USERS for mode in ('direct', 'replay', 'reverse', 'retained')} | {'1_eager_layout', '6_eager_layout'}
    require(set(warmups) == expected_warmups and set(profiles) == {76, 121}, 'Missing/extra untimed output evidence')
    runs = []; comparisons = []; anchors = {i: warmups[f'{i}_direct'] for i in USERS}
    for key, run in warmups.items():
        i = int(key.split('_')[0]); eager = key.endswith('_eager_layout')
        # Direct warmup occurs before all eight graphs exist; no final-owner
        # inclusion requirement there. Every reported owner is still checked.
        final_graphs = not eager and not key.endswith(('_direct', '_replay'))
        runs.append((key, run, i, required_supplied if final_graphs else (), required_graph_logits if final_graphs else ()))
    for row in requests:
        i = row['cohort_index']
        require(row['variant'] == ARMS[phase] and row['prompt_tokens'] == USERS[i], 'Timed arm/request identity differs')
        require('elapsed_s' in row['run'], 'Missing measured request duration')
        runs.append((f"timed_{row['repeat']}_{i}", row['run'], i, required_supplied, required_graph_logits))
    for length, run in profiles.items():
        i = 1 if length == 76 else 6
        runs.append((f'profile_{length}', run, i, required_supplied, required_graph_logits))
    output_checks = []; validities = []
    for name, run, i, supplied, logits in runs:
        output_checks.append(dict(name=name, **validate_output(run, by_user[i], vocab_size, required_supplied=set(supplied) | set(required_immutable), required_graph_logits=logits)))
        validities.append(dict(name=name, **sequence_validity(run, constraints, root_prefix)))
        if name != f'{i}_direct':
            comparisons.append(dict(name=name, cohort_index=i, **compare_outputs(anchors[i], run)))
    held = [p for i in USERS for p in warmups[f'{i}_retained']['saved_score_storage_ptrs']]
    require(len(set(held)) == 20, 'Retained requests share score storage')
    for i in USERS:
        require(warmups[f'{i}_replay']['saved_score_storage_ptrs'] == warmups[f'{i}_retained']['saved_score_storage_ptrs'], 'Retained output owner replaced')
    return dict(phase=phase, arm=ARMS[phase], output_checks=output_checks, validity=validities,
                comparisons=comparisons, same_arm_exact=all(c['exact'] for c in comparisons),
                sum_exact=all(c['five_step_fp32_sum_exact'] for c in output_checks),
                validity_passed=all(c['passed'] for c in validities), retained_owners=20)


def paired_timing(phases):
    require(set(phases) == set(range(4)), 'Four ABBA request partitions required')
    joined = {}
    for phase, rows in phases.items():
        expected = {(repeat, i) for repeat in range((phase // 2) * 3, (phase // 2) * 3 + 3) for i in USERS}
        require(len(rows) == 12 and {(r['repeat'], r['cohort_index']) for r in rows} == expected, 'Incomplete/duplicate timed partition')
        for row in rows:
            require(row['variant'] == ARMS[phase] and row['prompt_tokens'] == USERS[row['cohort_index']], 'Wrong timing arm/length')
            value = row['run']['elapsed_s']
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid measured duration')
            key = row['repeat'], row['cohort_index']
            cell = joined.setdefault(key, {})
            require(ARMS[phase] not in cell, 'Duplicate timing arm')
            cell[ARMS[phase]] = row['run']
    require(len(joined) == 24 and all(set(v) == set(ARMS) for v in joined.values()), 'Incomplete paired timing')
    strata = {'overall': list(joined)}
    strata.update({f'length_{length}': [k for k in joined if USERS[k[1]] == length] for length in (76, 121)})
    strata.update({f'half_{half}': [k for k in joined if k[0] // 3 == half] for half in (0, 1)})
    strata.update({f'length_{length}_half_{half}': [k for k in joined if USERS[k[1]] == length and k[0] // 3 == half]
                   for length in (76, 121) for half in (0, 1)})
    results = {}
    for name, keys in strata.items():
        result = {}
        for arm in ARMS[:2]:
            values = [joined[k][arm]['elapsed_s'] * 1000 for k in keys]
            result[arm] = dict(n=len(values), median_ms=statistics.median(values), p95_ms=statistics.quantiles(values, n=100, method='inclusive')[94])
        base, candidate = result[ARMS[0]], result[ARMS[1]]
        result['median_change_percent'] = 100 * (candidate['median_ms'] / base['median_ms'] - 1)
        result['p95_change_percent'] = 100 * (candidate['p95_ms'] / base['p95_ms'] - 1)
        result['passed'] = candidate['median_ms'] <= 0.95 * base['median_ms'] and candidate['p95_ms'] <= base['p95_ms']
        results[name] = result
    comparisons = [dict(repeat=k[0], cohort_index=k[1], **compare_outputs(v[ARMS[0]], v[ARMS[1]])) for k, v in sorted(joined.items())]
    return dict(strata=results, performance_passed=all(v['passed'] for v in results.values()),
                cross_arm=comparisons, cross_arm_exact=all(v['exact'] for v in comparisons),
                independent_users=4, paired_requests=24, population_equivalence_established=False,
                full500_eligibility_certified=False)
