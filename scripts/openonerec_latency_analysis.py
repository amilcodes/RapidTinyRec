"""Summarize openonerec_latency.py runs: per budget and policy, latency, certification and exactness vs beam.

usage: python scripts/openonerec_latency_analysis.py RUN.jsonl [RUN2.jsonl ...]
The exact top-K of a request is taken from the largest budget that certifies it (else the HF reference run's
exact list when the run carried --ref). Recall@K of a decoder = share of the exact top-K it returns.
"""
import json
import sys

import numpy as np


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def exact_top(r, K):
    for name in reversed(list(r['cfg'])):
        c = r['cfg'][name]
        if c.get(f'cert@{K}'):
            return [tuple(t) for t in c['top32'][:K]]
    return None


def summarize(path):
    R = [json.loads(l) for l in open(path) if l.strip()]
    if not R:
        return None
    cfgs = list(R[0]['cfg'])
    out = dict(file=path, n=len(R), prompt_p50=q([r['prompt_tokens'] for r in R], 50),
               prefill_ms_p50=q([r['prefill_ms'] for r in R], 50), prefill_ms_p90=q([r['prefill_ms'] for r in R], 90),
               ref_overlap=float(np.mean([r['ref_top10_overlap'] for r in R if 'ref_top10_overlap' in r] or [float('nan')])),
               ref_score_maxabs=max([r['ref_score_maxabs'] for r in R if r.get('ref_score_maxabs') is not None] or [None],
                                    key=lambda x: -1 if x is None else x),
               hf_rescore_maxabs=max([r['hf_rescore_maxabs'] for r in R if 'hf_rescore_maxabs' in r] or [None],
                                     key=lambda x: -1 if x is None else x),
               cfg={}, ladder={})
    ex = {K: [exact_top(r, K) for r in R] for K in (10, 32)}
    out['exact_known'] = {K: float(np.mean([e is not None for e in ex[K]])) for K in ex}
    for nm in cfgs:
        ms = [r['cfg'][nm]['ms'] for r in R]
        d = dict(ms_p50=q(ms, 50), ms_p90=q(ms, 90), ms_p99=q(ms, 99), ms_mean=float(np.mean(ms)),
                 total_p50=q([r['cfg'][nm]['ms'] + r['prefill_ms'] for r in R], 50),
                 **{f'cert@{k}': float(np.mean([r['cfg'][nm][f'cert@{k}'] for r in R])) for k in (1, 10, 32)})
        for K in (10, 32):
            rec = [len(set(map(tuple, r['cfg'][nm]['top32'][:K])) & set(e)) / K for r, e in zip(R, ex[K]) if e]
            d[f'recall@{K}'] = float(np.mean(rec)) if rec else float('nan')
            d[f'exact@{K}'] = float(np.mean([x == 1.0 for x in rec])) if rec else float('nan')
        for K in (10, 32):
            d[f'hit@{K}'] = float(np.mean([bool(set(map(tuple, r['cfg'][nm]['top32'][:K])) & set(map(tuple, r['targets'])))
                                           for r in R]))
        out['cfg'][nm] = d
    # numerical ties: where a certified budget's top-10 differs from the largest budget's, how far below the
    # reference tenth score are the swapped-in items (reference scores; items outside its top-32 count as > gap)
    big = cfgs[-1]
    for nm in cfgs[:-1]:
        gaps, ndiff, ncert = [], 0, 0
        for r in R:
            c, b = r['cfg'][nm], r['cfg'][big]
            if not (c.get('cert@10') and b.get('cert@10')):
                continue
            ncert += 1
            ref = {tuple(t): sc for t, sc in zip(b['top32'], b['scores32'])}
            t10 = b['scores32'][9]
            extra = [tuple(t) for t in c['top32'][:10] if tuple(t) not in set(map(tuple, b['top32'][:10]))]
            if extra:
                ndiff += 1
                gaps += [t10 - ref.get(t, b['scores32'][-1]) for t in extra]
        out['cfg'][nm]['tie_diff_share'] = ndiff / max(1, ncert)
        out['cfg'][nm]['tie_gap_max'] = max(gaps) if gaps else 0.0
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(R), size=(2000, len(R)))
    for B in ('32/32', '128/128'):
        for K in (10, 32):
            if B not in R[0]['cfg']:
                continue
            hit = lambda nm: np.array([bool(set(map(tuple, r['cfg'][nm]['top32'][:K])) & set(map(tuple, r['targets'])))
                                       for r in R], dtype=float)
            diff = hit(big) - hit(B)
            bs = diff[idx].mean(1)
            out['cfg'][B][f'hit{K}_exact_minus'] = [float(diff.mean()), q(bs, 2.5), q(bs, 97.5)]
    for key in R[0].get('ladder', {}):
        v = [r['ladder'][key] for r in R]
        ms = [x['ms'] for x in v]
        out['ladder'][key] = dict(ms_p50=q(ms, 50), ms_p90=q(ms, 90), ms_p99=q(ms, 99), ms_mean=float(np.mean(ms)),
                                  cert=float(np.mean([x['cert'] for x in v])),
                                  rung_hist=np.bincount([x['rung'] for x in v]).tolist())
    return out


def table(s):
    print(f"\n## {s['file']}  n={s['n']}  prompt p50 {s['prompt_p50']:.0f} tok  prefill p50/p90 "
          f"{s['prefill_ms_p50']:.2f}/{s['prefill_ms_p90']:.2f} ms  exact known @10/@32 "
          f"{s['exact_known'][10]:.3f}/{s['exact_known'][32]:.3f}")
    print(f"agreement with HF reference: top-10 overlap {s['ref_overlap']:.4f}, score maxabs {s['ref_score_maxabs']}, "
          f"HF re-score maxabs {s['hf_rescore_maxabs']}")
    print('| budget A1/M | decode ms p50 | p90 | p99 | total p50 | cert@1 | cert@10 | cert@32 | recall@10 | exact@10 | recall@32 | hit@10 | hit@32 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for nm, d in s['cfg'].items():
        print(f"| {nm} | {d['ms_p50']:.2f} | {d['ms_p90']:.2f} | {d['ms_p99']:.2f} | {d['total_p50']:.2f} | "
              f"{d['cert@1']:.3f} | {d['cert@10']:.3f} | {d['cert@32']:.3f} | {d['recall@10']:.3f} | {d['exact@10']:.3f} | "
              f"{d['recall@32']:.3f} | {d['hit@10']:.3f} | {d['hit@32']:.3f} |")
    print('ties (certified budgets vs largest): ' + ', '.join(
        f"{nm} {d['tie_diff_share']:.2f}/{d['tie_gap_max']:.3f}" for nm, d in s['cfg'].items() if 'tie_gap_max' in d))
    print('exact (largest budget) minus beam hit@K [95% CI]: ' + ', '.join(
        f"{nm} @{K} {d[f'hit{K}_exact_minus'][0]:+.3f} [{d[f'hit{K}_exact_minus'][1]:+.3f}, {d[f'hit{K}_exact_minus'][2]:+.3f}]"
        for nm, d in s['cfg'].items() for K in (10, 32) if f'hit{K}_exact_minus' in d))
    print('| policy @K (wall clock, incl. host checks) | ms p50 | p90 | p99 | mean | certified | stopped at rung |')
    print('|---|---:|---:|---:|---:|---:|---|')
    for k, d in s['ladder'].items():
        print(f"| {k} | {d['ms_p50']:.2f} | {d['ms_p90']:.2f} | {d['ms_p99']:.2f} | {d['ms_mean']:.2f} | {d['cert']:.3f} | "
              f"{d['rung_hist']} |")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        s = summarize(p)
        if s:
            table(s)
            json.dump(s, open(p + '.summary.json', 'w'), indent=1)
