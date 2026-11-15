"""Tables for the review experiments (scripts/review_job.sh): controlled comparison, verification, scaling.

usage: python3 scripts/review_analysis.py RUNS_DIR [--json OUT.json]
RUNS_DIR holds rv_ctlA, rv_ctlB, rv_ver1..3, rv_sclW, rv_sclL, rv_sclF (oscar.sh fetch) and baseEV (full-depth
reference lists). Everything missing is skipped.
"""
import argparse
import json
import random
from pathlib import Path


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def lat(r):
    if not r or 'latency' not in r:
        return None
    d = r['latency']['device_only']
    return d['median_ms'], d['p95_ms']


def boot(x, y, m, n=2000):
    d = [b[m] - a[m] for a, b in zip(x, y)]
    mu = sum(d) / len(d)
    random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(n))
    return 100 * mu, 100 * bs[int(0.025 * n)], 100 * bs[int(0.975 * n) - 1]


def agree(t1, t2, k=10):
    n = len(t1)
    top1 = sum(tuple(a[0]) == tuple(b[0]) for a, b in zip(t1, t2)) / n
    ov = sum(len({tuple(x) for x in a[:k]} & {tuple(x) for x in b[:k]}) / k for a, b in zip(t1, t2)) / n
    same = sum({tuple(x) for x in a[:k]} == {tuple(x) for x in b[:k]} for a, b in zip(t1, t2)) / n
    return top1, ov, same


def runinfo(root, job, name):
    d = root / job / name
    r = load(d / 'results.json')
    q = load(d / 'quality_users.json')
    ql = load(d / 'quality_lists.json')
    lu = load(d / 'latency_users.json')
    return {'r': r, 'q': q, 'ql': ql, 'lu': lu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = Path(a.runs)
    out = {}
    M = ['HR@1', 'HR@10', 'NDCG@10', 'HR@50']

    # ---------------- controlled comparison
    rows = [('rv_ctlA', 'X1_beam_cbtrie'), ('rv_ctlA', 'X2_cert_full'), ('rv_ctlA', 'X3_cert_c22'),
            ('rv_ctlA', 'X4_cert_18_26'), ('rv_ctlB', 'Y1_beam_cb'), ('rv_ctlB', 'Y2_cert_cb'),
            ('rv_ctlB', 'Y3_cert_cb_c22'), ('rv_ctlB', 'Z1_beam_full')]
    R = {name: runinfo(root, job, name) for job, name in rows}
    print('\n== controlled comparison (same allocation per objective, FP16)')
    print(f"{'config':18s} {'median':>7s} {'p95':>6s}  " + ' '.join(f'{m:>8s}' for m in M) +
          '  cert@10  users/s B=16/32/64')
    ctl = {}
    for job, name in rows:
        x = R[name]
        if not x['r']:
            print(f'{name:18s} (missing)')
            continue
        l = lat(x['r'])
        qm = (x['r'].get('quality') or {}).get('metrics', {})
        cr = ((x['r'].get('quality') or {}).get('certified_rate') or (x['r'].get('latency') or {}).get('certified_rate') or {})
        tp = x['r'].get('throughput') or {}
        tps = '/'.join(f"{tp[b]['users_per_s']:.0f}" for b in ('16', '32', '64') if b in tp)
        print(f"{name:18s} {l[0] if l else float('nan'):7.2f} {l[1] if l else float('nan'):6.2f}  " +
              ' '.join(f"{qm.get(m, float('nan')):8.4f}" for m in M) + f"  {cr.get('top10', '-')!s:7s}  {tps}")
        ctl[name] = {'median_ms': l[0] if l else None, 'p95_ms': l[1] if l else None, 'metrics': qm,
                     'cert': cr, 'throughput': {b: tp[b]['users_per_s'] for b in tp}}
    out['controlled'] = ctl
    pairs = [('X1_beam_cbtrie', 'X2_cert_full'), ('X2_cert_full', 'X3_cert_c22'), ('X2_cert_full', 'X4_cert_18_26'),
             ('X1_beam_cbtrie', 'X4_cert_18_26'), ('Y1_beam_cb', 'Y2_cert_cb'), ('Y2_cert_cb', 'Y3_cert_cb_c22')]
    print('\n   paired (second minus first), points; ranking agreement of the top-10 lists')
    out['paired'] = {}
    for p1, p2 in pairs:
        if not (R[p1]['q'] and R[p2]['q']):
            continue
        u1, u2 = R[p1]['q']['per_user'], R[p2]['q']['per_user']
        ds = {m: boot(u1, u2, m) for m in ('HR@10', 'NDCG@10', 'HR@50')}
        ag = agree(R[p1]['q']['top10'], R[p2]['q']['top10'])
        print(f"   {p2:16s} vs {p1:16s} " + '  '.join(f"{m} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for m, v in ds.items()) +
              f"   top1 {ag[0]:.3f} top10-overlap {ag[1]:.3f} same-set {ag[2]:.3f}")
        out['paired'][f'{p2}_vs_{p1}'] = {'delta': ds, 'agreement': ag}

    # ---------------- verification
    print('\n== verification against scoring every item under the same objective')
    V = {n: runinfo(root, j, n) for j, n in [('rv_ver1', 'V1_full_M16'), ('rv_ver1', 'V2_full_all'),
                                             ('rv_ver2', 'V3_br_M16'), ('rv_ver2', 'V4_br_all'),
                                             ('rv_ver3', 'V5_full_fp32')]}
    out['verify'] = {}
    for mine, ref, label in [('V1_full_M16', 'V2_full_all', 'full depth'), ('V3_br_M16', 'V4_br_all', 'bridged')]:
        a_, b_ = V[mine], V[ref]
        if not (a_['ql'] and b_['ql']):
            print(f'   {label}: missing lists')
            continue
        ks = a_['ql']['ks']
        j10 = ks.index(10)
        n = cert = bad = bad_tie = 0
        worst = 0.0
        for t, sc, c, tr, scr in zip(a_['ql']['top50'], a_['ql']['scores50'], a_['ql']['cert'],
                                     b_['ql']['top50'], b_['ql']['scores50']):
            n += 1
            if not c[j10]:
                continue
            cert += 1
            mine10 = {tuple(x) for x in t[:10]}
            ref10 = [tuple(x) for x in tr[:10]]
            if mine10 != set(ref10):
                refsc = dict(zip(map(tuple, tr), scr))
                th = sc[9]                              # my 10th score
                gap = max(refsc[g] - th for g in ref10 if g not in mine10)
                worst = max(worst, gap)
                if gap > 5e-2:
                    bad += 1
                else:
                    bad_tie += 1
        print(f"   {label:10s}: {cert}/{n} users certified at K=10; certified top-10 differs from exhaustive for "
              f"{bad + bad_tie} ({bad_tie} within FP16 tolerance 0.05, {bad} beyond); worst gap {worst:.3f}")
        out['verify'][label] = {'users': n, 'certified10': cert, 'differs': bad + bad_tie, 'beyond_tol': bad,
                                'worst_gap': worst}
    for name in ('V1_full_M16', 'V5_full_fp32'):
        r = V[name]['r']
        st = (r or {}).get('latency', {}).get('vs_exhaustive')
        if st:
            print(f"   {name}: scorer check on {st['n']} cohort users: certified top-10 users {st['cert_top10_users']}, "
                  f"violations {st['cert_violations']}, top-1 match {st['top1_match']:.3f}, recall@10 {st['search_recall@10']:.3f}")
            out['verify'][name] = st
    base = load(root / 'baseEV' / 'ev_test' / 'quality_users.json')
    if base and V['V2_full_all']['q'] and V['V4_br_all']['q']:
        ag = agree(V['V2_full_all']['q']['top10'], V['V4_br_all']['q']['top10'])
        print(f"   bridged objective vs full-depth objective (both exhaustive): top1 {ag[0]:.3f} top10-overlap {ag[1]:.3f}")
        out['verify']['bridged_vs_full_objective'] = ag

    # ---------------- scaling
    print('\n== serving boundary (latency only; quality meaningless on synthetic items)')
    out['scale'] = {}
    for job in ('rv_sclW', 'rv_sclL', 'rv_sclF'):
        d = root / job
        if not d.exists():
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and (p / 'results.json').exists()):
            r = load(sub / 'results.json')
            l = lat(r)
            cr = (r.get('latency') or {}).get('certified_rate') or {}
            lu = load(sub / 'latency_users.json') or []
            Ls = sorted({u['L'] for u in lu})
            print(f"   {sub.name:16s} median {l[0]:6.2f}  p95 {l[1]:6.2f}  cert@10 {cr.get('top10', '-')!s:6s} "
                  f"prompt {Ls[0] if Ls else '?'}-{Ls[-1] if Ls else '?'} tokens  S={r['meta'].get('bonus', {}).get('S', '')}")
            out['scale'][sub.name] = {'median_ms': l[0], 'p95_ms': l[1], 'cert': cr,
                                      'prompt_tokens': [Ls[0], Ls[-1]] if Ls else None}
    # p99 with a fallback rerun: users whose top-10 is not certified at M=16 are re-run at M=32
    f16, f32 = runinfo(root, 'rv_sclF', 'F_1M_M16'), runinfo(root, 'rv_sclF', 'F_1M_M32')
    if f16['lu'] and f32['lu']:
        ms32 = {u['row']: u['ms'] for u in f32['lu']}
        j10 = [k for k in (1, 5, 10, 20, 50)].index(10)
        tot = sorted(u['ms'] + (ms32[u['row']] if u['cert'] and not u['cert'][j10] else 0.0) for u in f16['lu'])
        nfb = sum(1 for u in f16['lu'] if u['cert'] and not u['cert'][j10])
        q = lambda p: tot[min(len(tot) - 1, int(p * len(tot)))]
        print(f"   1M SIDs, M=16 with fallback to M=32 for uncertified top-10: {nfb}/{len(tot)} reruns; "
              f"median {q(0.5):.2f}  p95 {q(0.95):.2f}  p99 {q(0.99):.2f}  max {tot[-1]:.2f} ms")
        out['scale']['fallback_1M'] = {'reruns': nfb, 'n': len(tot), 'p50': q(0.5), 'p95': q(0.95), 'p99': q(0.99),
                                       'max': tot[-1]}
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
