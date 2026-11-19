"""Summarize a flashrec_job.sh run: latency, quality, beam agreement, throughput (FlashRec vs ours, one allocation).

usage: python scripts/compare_flashrec.py RUN_DIR [THROUGHPUT_RUN_DIR] [--json OUT.json]
"""
import argparse, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('tp_run', nargs='?'); ap.add_argument('--json')
_a = ap.parse_args()
R = Path(_a.run); TP = Path(_a.tp_run) if _a.tp_run else None; jout = _a.json


def load(d, f='results.json'):
    p = d / f
    return json.loads(p.read_text()) if p.exists() else None


def agree(a, b):
    """per-user top-1 match, top-10 / top-50 set overlap between two latency_users.json (same cohort order)."""
    A = {r['row']: r for r in a}; B = {r['row']: r for r in b}
    rows = [r for r in A if r in B]
    t1 = sum(A[r]['sids'][0] == B[r]['sids'][0] for r in rows) / len(rows)
    ov = lambda k: sum(len({tuple(x) for x in A[r]['sids'][:k]} & {tuple(x) for x in B[r]['sids'][:k]}) / k
                       for r in rows) / len(rows)
    same10 = sum({tuple(x) for x in A[r]['sids'][:10]} == {tuple(x) for x in B[r]['sids'][:10]} for r in rows) / len(rows)
    return {'users': len(rows), 'top1_match': t1, 'top10_overlap': ov(10), 'top10_same_set': same10, 'top50_overlap': ov(50)}


def paired(a, b, m='HR@10', B=2000):
    a = a['per_user']; b = b['per_user']; n = len(a); assert n == len(b)
    d = [y[m] - x[m] for x, y in zip(a, b)]
    rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {'A': sum(x[m] for x in a) / n, 'B': sum(y[m] for y in b) / n, 'B-A_pt': 100 * sum(d) / n,
            'ci_pt': [100 * bs[int(.025 * B)], 100 * bs[int(.975 * B) - 1]]}


out = {'latency': {}, 'quality': {}, 'agreement': {}, 'paired': {}, 'throughput': {}}
for d in sorted(R.iterdir()):
    if not d.is_dir():
        continue
    r = load(d)
    if r is None:
        continue
    lat = r.get('latency') or {}
    if 'api_wall' in lat:
        out['latency'][d.name] = {'median_ms': lat['api_wall']['median_ms'], 'p95_ms': lat['api_wall']['p95_ms'],
                                  'boundary': 'engine.generate() wall', 'HR@10_500': lat['metrics_500']['HR@10']}
    elif 'device_only' in lat:
        out['latency'][d.name] = {'median_ms': lat['device_only']['median_ms'], 'p95_ms': lat['device_only']['p95_ms'],
                                  'median_with_d2h_ms': lat['with_d2h']['median_ms'], 'boundary': 'graph replay (+ D2H)',
                                  'HR@10_500': lat['metrics_500']['HR@10']}
    q = r.get('quality') or {}
    if q.get('metrics'):
        out['quality'][d.name] = {k: q['metrics'][k] for k in ('HR@1', 'HR@10', 'NDCG@10', 'HR@50')} | \
            {k: q[k] for k in ('invalid_sids', 'beams_min') if k in q}
ref = R / 'O1_beam_cb'
for d in sorted(R.glob('F_*')):
    a, b = load(ref, 'latency_users.json'), load(d, 'latency_users.json')
    if a and b:
        out['agreement'][f'{d.name} vs O1_beam_cb'] = agree(a, b)
    qa, qb = load(ref, 'quality_users.json'), load(d, 'quality_users.json')
    if qa and qb:
        out['paired'][f'{d.name} - O1_beam_cb'] = {m: paired(qa, qb, m) for m in ('HR@10', 'NDCG@10')}
if TP:
    for d in sorted(TP.iterdir()):
        r = load(d) if d.is_dir() else None
        if r and r.get('throughput'):
            t = r['throughput']
            out['throughput'][d.name] = {str(k): (v['users_per_s'] if isinstance(v, dict) else v) for k, v in
                                         (t.items() if isinstance(t, dict) else enumerate(t))}
print(json.dumps(out, indent=1))
if jout:
    json.dump(out, open(jout, 'w'), indent=1)
