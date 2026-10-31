"""Independently audit a fetched complete MiniOneRec paired benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--token-map', type=Path)
    args = parser.parse_args()
    root = args.results
    manifest = json.loads((root / 'manifest.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    stored_hash = manifest['manifest_sha256']
    payload = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == stored_hash
    expected = json.loads((Path(__file__).parent / 'expected.json').read_text())
    cohort_hash = hashlib.sha256(json.dumps(manifest['users'], sort_keys=True).encode()).hexdigest()
    assert cohort_hash == expected['users_sha256'] == manifest['cohort_sha256']
    assert len(manifest['users']) == 500
    paths = sorted((root / 'users').glob('*.json'))
    assert len(paths) == 500 and summary['complete']
    records = [json.loads(p.read_text()) for p in paths]
    assert len({r['user_id'] for r in records}) == 500
    for i, r in enumerate(records):
        assert r['cohort_index'] == i
        assert r['manifest_sha256'] == stored_hash
        assert r['user_id'] == manifest['users'][i]['user_id']
        assert r['source_row_index'] == manifest['users'][i]['source_row_index']
        assert r['prompt_tokens'] == manifest['users'][i]['prompt_tokens']
        assert r['order'] == (['baseline', 'shared'] if i % 2 == 0 else ['shared', 'baseline'])
        for v in ('baseline', 'shared'):
            run = r['runs'][v]
            assert len(run['token_ids']) == len(run['sequence_logp']) == len(run['transition_logp']) == 50
            assert len({tuple(s) for s in run['token_ids']}) == 50
            assert math.isfinite(run['elapsed_s']) and run['elapsed_s'] > 0
            assert all(math.isfinite(x) for x in run['sequence_logp'])
            assert all(math.isfinite(x) for scores in run['transition_logp'] for x in scores)

    latency = {}
    for v in ('baseline', 'shared'):
        values = [r['runs'][v]['elapsed_s'] for r in records]
        latency[v] = {'n': len(values), 'median_s': statistics.median(values),
                      'p95_s': statistics.quantiles(values, n=100, method='inclusive')[94]}
        for key in ('median_s', 'p95_s'):
            assert math.isclose(latency[v][key], summary[v][key], rel_tol=1e-12)
    equal_rank = equal_set = equal_top1 = equal_scores = equal_transitions = positions = 0
    overlaps = []
    mismatch_indices = []
    score_bit_checks = {}
    for field in ('sequence_logp', 'transition_logp'):
        matching_users = matching_values = total_values = 0
        max_abs_difference = 0.0
        for r in records:
            a, b = (r['runs'][v][field] for v in ('baseline', 'shared'))
            if field == 'transition_logp':
                assert [len(x) for x in a] == [len(x) for x in b]
                a, b = ([value for row in array for value in row] for array in (a, b))
            assert len(a) == len(b)
            bits = [struct.pack('!f', x) == struct.pack('!f', y) for x, y in zip(a, b)]
            matching_users += all(bits)
            matching_values += sum(bits)
            total_values += len(bits)
            max_abs_difference = max(max_abs_difference, max(abs(x-y) for x,y in zip(a,b)))
        score_bit_checks[field] = {'bitwise_equal_users': matching_users,
            'bitwise_equal_float32_values': matching_values, 'total_values': total_values,
            'max_absolute_difference_in_rank_order': max_abs_difference}
    for r in records:
        a, b = (r['runs'][v] for v in ('baseline', 'shared'))
        ra, rb = a['token_ids'], b['token_ids']
        sa, sb = {tuple(x) for x in ra}, {tuple(x) for x in rb}
        equal_rank += ra == rb
        equal_set += sa == sb
        equal_top1 += ra[0] == rb[0]
        equal_scores += a['sequence_logp'] == b['sequence_logp']
        equal_transitions += a['transition_logp'] == b['transition_logp']
        positions += sum(x == y for x, y in zip(ra, rb))
        overlaps.append(len(sa & sb))
        if ra != rb:
            mismatch_indices.append(r['cohort_index'])
    checks = {'all50_rankings_match_users': equal_rank, 'same_top50_set_users': equal_set,
              'same_top1_users': equal_top1, 'same_sequence_scores_users': equal_scores,
              'same_transition_scores_users': equal_transitions,
              'matching_rank_positions': positions, 'total_rank_positions': 25000,
              'mismatching_cohort_indices': mismatch_indices}
    for k, value in checks.items():
        assert value == summary[k], (k, value, summary[k])
    report = {'audit_passed': True, 'results': str(root.resolve()),
              'hardware': manifest['hardware'], 'dtype': manifest['dtype'],
              'experiment': manifest.get('experiment', 'shared_prefill'),
              'variant_definitions': manifest.get('variant_definitions', {
                  'baseline': {'shared_prefill': False, 'constraint_processor': 'ConstrainedLogitsProcessor'},
                  'shared': {'shared_prefill': True, 'constraint_processor': 'ConstrainedLogitsProcessor'}}),
              'versions': manifest['versions'],
              'cuda_runtime': manifest['cuda_runtime'], 'cohort_sha256': cohort_hash,
              'manifest_sha256': stored_hash, 'users': 500, 'measured_requests': 1000,
              'first_variant_counts': {v: sum(r['order'][0] == v for r in records)
                                       for v in ('baseline', 'shared')},
              'latency': latency,
              'median_speedup': latency['baseline']['median_s'] / latency['shared']['median_s'],
              'p95_speedup': latency['baseline']['p95_s'] / latency['shared']['p95_s'],
              'ranking_checks': checks,
              'score_bit_checks': score_bit_checks,
              'mean_top50_overlap': statistics.mean(overlaps),
              'min_top50_overlap': min(overlaps),
              'scope': manifest['scope']}
    if args.token_map:
        assert hashlib.sha256(args.token_map.read_bytes()).hexdigest() == expected['release_sha256']['Office_ckpt/added_tokens.json']
        token_map = json.loads(args.token_map.read_text())
        target_ranks = {v: [] for v in ('baseline', 'shared')}
        for r in records:
            target_tokens = [token_map[t] for t in re.findall(r'<[^>]+>', r['target_sid'])]
            assert len(target_tokens) == 3
            for v in target_ranks:
                ranks = [i + 1 for i, tokens in enumerate(r['runs'][v]['token_ids'])
                         if tokens[:3] == target_tokens]
                target_ranks[v].append(ranks[0] if ranks else None)
        report['target_sid_metrics'] = {
            v: {str(k): {'hits': sum(rank is not None and rank <= k for rank in ranks),
                         'hit_rate': sum(rank is not None and rank <= k for rank in ranks) / 500,
                         'ndcg': sum(1 / math.log2(rank + 1) for rank in ranks
                                     if rank is not None and rank <= k) / 500}
                for k in (1, 5, 10, 20, 50)}
            for v, ranks in target_ranks.items()}
        report['paired_target_hit_changes'] = {
            str(k): {
                'baseline_only': sum(a is not None and a <= k and not (c is not None and c <= k)
                                     for a, c in zip(target_ranks['baseline'], target_ranks['shared'])),
                'shared_only': sum(c is not None and c <= k and not (a is not None and a <= k)
                                   for a, c in zip(target_ranks['baseline'], target_ranks['shared']))}
            for k in (1, 5, 10, 20, 50)}
        report['target_metric_scope'] = 'Single held-out semantic-ID target per selected user; SID-level hit rate and NDCG, not an item-level collision disambiguation or statistical superiority claim.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
