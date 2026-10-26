"""Paired held-out SID quality; numerical fidelity is a separate audit.

No score-equality or production-equivalence decision is made here. Repeated
requests must not be counted as independent users. Invalid or duplicate outputs
keep their original rank positions; removing them would inflate quality.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

CUTOFFS = (1, 5, 10, 20, 50)
METRICS = ('hit_rate', 'ndcg', 'mrr')
VARIANTS = ('control', 'candidate')
BOOTSTRAP_SEED = 20261026
BOOTSTRAP_SAMPLES = 10000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sid(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 3 and
             all(type(x) is int and x >= 0 for x in value), 'SID must contain three nonnegative token IDs')
    return tuple(value)


def _vector(rank):
    return [value for k in CUTOFFS for value in (
        float(rank is not None and rank <= k),
        1 / math.log2(rank + 1) if rank is not None and rank <= k else 0.,
        1 / rank if rank is not None and rank <= k else 0.)]


def _named(vector):
    return {str(k): dict(zip(METRICS, vector[i*3:i*3+3])) for i, k in enumerate(CUTOFFS)}


def _mean(vectors):
    return [math.fsum(v[j] for v in vectors) / len(vectors) for j in range(15)]


def _quantile(sorted_values, q):
    x = (len(sorted_values) - 1) * q
    lo = int(x)
    return sorted_values[lo] + (sorted_values[min(lo + 1, len(sorted_values)-1)] - sorted_values[lo]) * (x-lo)


def _intervals(deltas, samples, seed):
    rng = random.Random(seed)
    draws = [[] for _ in range(15)]
    for _ in range(samples):
        counts = Counter(rng.randrange(len(deltas)) for _ in deltas)
        for j in range(15):
            draws[j].append(math.fsum(deltas[i][j]*n for i,n in counts.items()) / len(deltas))
    return _named([{'lower': _quantile(sorted(v), .025), 'upper': _quantile(sorted(v), .975)} for v in draws])


def evaluate(users, observations, catalog_sids, eos_token_id, *, bootstrap_samples=BOOTSTRAP_SAMPLES,
             bootstrap_seed=BOOTSTRAP_SEED):
    """Inputs use target_token_ids (3) and original five-token beam outputs (50).

    The caller must independently verify token-map/catalog/held-out provenance and
    score bits. This function validates pairing and produces descriptive quality
    and paired user-bootstrap intervals, never a promotion or noninferiority pass.
    """
    _require(type(bootstrap_samples) is int and bootstrap_samples >= 2, 'At least two bootstrap samples required')
    _require(type(bootstrap_seed) is int, 'Integer bootstrap seed required')
    _require(type(eos_token_id) is int and eos_token_id >= 0, 'Invalid EOS token')
    _require(bool(users), 'No users')
    catalog = {_sid(s) for s in catalog_sids}
    _require(bool(catalog), 'Empty catalog')
    user_map = {}
    for u in users:
        index = u['cohort_index']
        _require(type(index) is int and index not in user_map, 'Duplicate or invalid cohort index')
        _require(type(u['prompt_tokens']) is int and u['prompt_tokens'] > 0, 'Invalid prompt length')
        target = _sid(u['target_token_ids'])
        _require(target in catalog, 'Held-out target absent from catalog')
        user_map[index] = dict(u, target_token_ids=target)
    runs = defaultdict(dict)
    for row in observations:
        index, variant, repeat = row['cohort_index'], row['variant'], row['repeat']
        _require(type(index) is int and index in user_map and variant in VARIANTS, 'Unknown user or variant')
        _require(type(repeat) is int and repeat >= 0, 'Invalid repeat')
        key = (variant, repeat)
        _require(key not in runs[index], 'Duplicate user/variant/repeat')
        tokens = row['token_ids']
        _require(isinstance(tokens, list) and len(tokens) == 50, 'Exactly 50 original rank positions required')
        _require(all(isinstance(seq, list) and all(type(x) is int and x >= 0 for x in seq) for seq in tokens),
                 'Output tokens must be nonnegative integer lists')
        runs[index][key] = tokens
    _require(set(runs) == set(user_map), 'Missing user observations')
    rows = []
    all_catalog = {v: set() for v in VARIANTS}
    for index in sorted(user_map):
        u = user_map[index]
        repeat_sets = {v: {r for variant,r in runs[index] if variant == v} for v in VARIANTS}
        _require(repeat_sets['control'] and repeat_sets['control'] == repeat_sets['candidate'], 'Unpaired repeats')
        first = min(repeat_sets['control'])
        result = dict(cohort_index=index, prompt_tokens=u['prompt_tokens'], repeats=len(repeat_sets['control']),
                      representative_repeat=first, variants={})
        for variant in VARIANTS:
            token_rows = runs[index][variant, first]
            stable = all(runs[index][variant,r] == token_rows for r in repeat_sets[variant])
            valid = [len(seq) == 5 and seq[3:] == [198, eos_token_id] and tuple(seq[:3]) in catalog for seq in token_rows]
            sids = [tuple(seq[:3]) if ok else None for seq,ok in zip(token_rows,valid)]
            good = [s for s in sids if s is not None]
            rank = next((i+1 for i,s in enumerate(sids) if s == u['target_token_ids']), None)
            all_catalog[variant].update(good)
            result['variants'][variant] = dict(target_rank=rank, valid_outputs=sum(valid),
                invalid_outputs=50-sum(valid), duplicate_valid_sids=len(good)-len(set(good)),
                stable_token_ranks_across_repeats=stable, metrics=_named(_vector(rank)))
        a, b = runs[index]['control', first], runs[index]['candidate', first]
        sa, sb = {tuple(x[:3]) for x in a}, {tuple(x[:3]) for x in b}
        result['fidelity'] = dict(identical_token_rank_positions=sum(x==y for x,y in zip(a,b)),
            compared_rank_positions=50, top1_token_match=a[0]==b[0],
            sid_set_jaccard=len(sa & sb)/len(sa | sb),
            note='Overlap includes original outputs; validity reported separately. No score-bit test here.')
        av = _vector(result['variants']['control']['target_rank'])
        bv = _vector(result['variants']['candidate']['target_rank'])
        result['paired_changes'] = _named([y-x for x,y in zip(av,bv)])
        rows.append(result)

    def aggregate(selected, seed):
        vectors = {v: [_vector(r['variants'][v]['target_rank']) for r in selected] for v in VARIANTS}
        deltas = [[y-x for x,y in zip(a,b)] for a,b in zip(vectors['control'], vectors['candidate'])]
        return dict(independent_users=len(selected),
            metrics={v: _named(_mean(vectors[v])) for v in VARIANTS},
            paired_mean_changes=_named(_mean(deltas)),
            paired_user_bootstrap_95pct=_intervals(deltas,bootstrap_samples,seed),
            changed_users=_named([dict(improved=sum(x[j]>0 for x in deltas),
                                      regressed=sum(x[j]<0 for x in deltas),
                                      unchanged=sum(x[j]==0 for x in deltas)) for j in range(15)]))

    strata = sorted({r['prompt_tokens'] for r in rows})
    result = dict(schema='paired-sid-quality-v1', primary_metrics=['ndcg@10','hit_rate@10'],
        overall=aggregate(rows, bootstrap_seed),
        by_prompt_length={str(n): aggregate([r for r in rows if r['prompt_tokens']==n], bootstrap_seed+n) for n in strata},
        per_user=rows, catalog_size=len(catalog),
        catalog_coverage={v: len(all_catalog[v])/len(catalog) for v in VARIANTS},
        output_validity={v: {key:sum(r['variants'][v][key] for r in rows) for key in
                            ('invalid_outputs','duplicate_valid_sids')} for v in VARIANTS},
        token_repeat_stability=all(r['variants'][v]['stable_token_ranks_across_repeats'] for r in rows for v in VARIANTS),
        bootstrap=dict(samples=bootstrap_samples,seed=bootstrap_seed,unit='unique user; paired variants resampled together',
            method='two-sided percentile 95%, linear interpolation; descriptive, unadjusted multiple comparisons'),
        limitations=['One held-out SID per user; no item-collision disambiguation.',
            'First matched repeat determines quality; all repeats checked for token-rank stability separately.',
            'Intervals describe sampling uncertainty on observed users; tiny/all-zero samples cannot exclude unseen harms.',
            'No numerical-fidelity, provenance, performance, production-equivalence or noninferiority gate is decided here.'],
        promotion=False, noninferiority_established=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    _require(not args.output.exists(),'Refusing to overwrite prior quality evidence')
    raw=args.input.read_bytes(); data=json.loads(raw)
    report=evaluate(data['users'],data['observations'],data['catalog_sids'],data['eos_token_id'])
    report['input_sha256']=hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
