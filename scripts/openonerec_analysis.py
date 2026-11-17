"""Summarize OpenOneRec certification-cost runs (scripts/openonerec_cert.py output).

usage: python scripts/openonerec_analysis.py RUN.jsonl [RUN2.jsonl ...]   (each file: one benchmark task)
Per normalization and K: share of requests certified, the instance-optimal number of first codes (N_A) and
two-code prefixes (N_P, and N_Pmulti = those with more than one item) that any exact method must expand, the rows
the best-first search actually expanded, beam search's recall of the exact top-K, and target hit rates.
"""
import json
import sys

import numpy as np


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        return None
    norms = [k for k in ('full', 'codebook', 'trie') if k in recs[0]]
    ks = sorted({int(k.split('@')[1]) for k in recs[0][norms[0]] if k.startswith('cert@')})
    out = dict(file=path, n=len(recs), prompt_tokens_p50=q([r['prompt_tokens'] for r in recs], 50),
               targets_per_req=float(np.mean([len(r['targets']) for r in recs])),
               targets_in_catalog=float(np.mean([r.get('targets_in_catalog', len(r['targets'])) / max(1, len(r['targets']))
                                                 for r in recs])),
               seconds_p50=q([r['seconds'] for r in recs], 50),
               tree_check_maxabs=max((r['tree_check_maxabs'] for r in recs if 'tree_check_maxabs' in r), default=None),
               norms={})
    for nm in norms:
        R = [r[nm] for r in recs]
        d = dict(expanded_a_p50=q([x['expanded_a'] for x in R], 50), expanded_a_p90=q([x['expanded_a'] for x in R], 90),
                 expanded_p_p50=q([x['expanded_p'] for x in R], 50), expanded_p_p90=q([x['expanded_p'] for x in R], 90),
                 hit_exact1=float(np.mean([x['hit_exact@1'] for x in R])),
                 hit_exact32=float(np.mean([x['hit_exact@32'] for x in R])), K={})
        for K in ks:
            c = [x for x in R if x.get(f'cert@{K}')]
            e = dict(cert=len(c) / len(R))
            for f in ('N_A', 'N_P', 'N_Pmulti'):
                v = [x[f'{f}@{K}'] for x in c if f'{f}@{K}' in x]
                e[f + '_p50'], e[f + '_p90'], e[f + '_max'] = q(v, 50), q(v, 90), (max(v) if v else None)
            d['K'][K] = e
        beams = sorted({int(k[4:].split('_')[0]) for k in R[0] if k.startswith('beam') and '_recall@' in k})
        for B in beams:
            for K in (1, 10, 32):
                v = [x[f'beam{B}_recall@{K}'] for x in R if f'beam{B}_recall@{K}' in x]
                if v:
                    d[f'beam{B}_recall@{K}'] = float(np.mean(v))
                    d[f'beam{B}_exact@{K}'] = float(np.mean([x == 1.0 for x in v]))
            d[f'hit_beam{B}@1'] = float(np.mean([x[f'hit_beam{B}@1'] for x in R]))
            d[f'hit_beam{B}@32'] = float(np.mean([x[f'hit_beam{B}@32'] for x in R]))
        out['norms'][nm] = d
    return out


def table(s):
    print(f"\n## {s['file']}  (n={s['n']}, prompt p50 {s['prompt_tokens_p50']:.0f} tok, {s['seconds_p50']:.1f} s/req, "
          f"targets/req {s['targets_per_req']:.1f}, in catalog {s['targets_in_catalog']:.3f}, "
          f"tree check {s['tree_check_maxabs']})")
    for nm, d in s['norms'].items():
        print(f"### {nm}: expanded first codes p50/p90 {d['expanded_a_p50']:.0f}/{d['expanded_a_p90']:.0f}, "
              f"prefixes {d['expanded_p_p50']:.0f}/{d['expanded_p_p90']:.0f}; hit@1/@32 exact "
              f"{d['hit_exact1']:.3f}/{d['hit_exact32']:.3f}, " +
              ', '.join(f"beam{B} {d.get(f'hit_beam{B}@1', float('nan')):.3f}/{d.get(f'hit_beam{B}@32', float('nan')):.3f}"
                        for B in (32, 128)))
        print('| K | cert | N_A p50 | N_A p90 | N_P p50 | N_P p90 | N_Pmulti p50 |')
        print('|---:|---:|---:|---:|---:|---:|---:|')
        for K, e in d['K'].items():
            print(f"| {K} | {e['cert']:.3f} | {e['N_A_p50']:.0f} | {e['N_A_p90']:.0f} | {e['N_P_p50']:.0f} | "
                  f"{e['N_P_p90']:.0f} | {e['N_Pmulti_p50']:.0f} |")
        print('beam recall of exact top-K (mean / share exact): ' + ', '.join(
            f"B{B}@{K} {d[f'beam{B}_recall@{K}']:.3f}/{d[f'beam{B}_exact@{K}']:.2f}"
            for B in (32, 128) for K in (1, 10, 32) if f'beam{B}_recall@{K}' in d))


if __name__ == '__main__':
    allsum = [s for s in (summarize(p) for p in sys.argv[1:]) if s]
    for s in allsum:
        table(s)
    for s in allsum:
        json.dump(s, open(s['file'] + '.summary.json', 'w'), indent=1)
