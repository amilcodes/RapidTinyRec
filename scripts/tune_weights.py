"""Tune depth-discounted scoring on the validation split, evaluate once on test.

usage: tune_weights.py VALID_DUMP_DIR TEST_DUMP_DIR [--out results.json]

score(item) = s(a) + wb s(b|a) + wc s(c|ab) + ws suffix   for each base rule
(full / codebook / trie; suffix only exists under full). Grid-search maximizes validation
NDCG@10 under exhaustive (= certified exact) search; the frozen choice is then scored on test
with paired-bootstrap CIs against the reference decode (full rule, beam 50, all weights 1).
"""
import argparse, itertools, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as S

WB = (0.5, 0.75, 1.0, 1.25)
WC = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
WS = (0.0, 0.1, 0.3, 0.5, 1.0)


def prep(dump):
    D, meta = S.load(dump)
    sid_list = [tuple(x) for x in meta['sid_tok'].tolist()]
    pos_of = {s: i for i, s in enumerate(sid_list)}
    tgt = torch.tensor([pos_of.get(tuple(t), -1) for t in D['targets']])
    terms = {r: S.terms(D, meta, r) for r in ('full', 'codebook', 'trie')}
    return D, meta, tgt, terms


def total(meta, t, wb, wc, ws):
    sa, sb, sc, suf = t
    u2 = (sa[..., None] + wb * sb).reshape(sa.shape[0], -1)[:, meta['sid_flat']]
    return u2 + wc * sc + ws * suf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('valid'); ap.add_argument('test'); ap.add_argument('--out')
    a = ap.parse_args()
    Dv, mv, tv, Tv = prep(a.valid)
    Dt, mt, tt, Tt = prep(a.test)
    grid = []
    for rule in ('full', 'codebook', 'trie'):
        for wb, wc in itertools.product(WB, WC):
            for ws in (WS if rule == 'full' else (0.0,)):
                m = S.summarize(S.metrics(S.exhaustive_rank(total(mv, Tv[rule], wb, wc, ws))[:, :20], tv))
                grid.append({'rule': rule, 'wb': wb, 'wc': wc, 'ws': ws, **{k: m[k] for k in ('HR@10', 'NDCG@10', 'HR@1')}})
    best = {}
    for rule in ('full', 'codebook', 'trie'):
        best[rule] = max((g for g in grid if g['rule'] == rule), key=lambda g: g['NDCG@10'])
    best['full-nosuffix'] = max((g for g in grid if g['rule'] == 'full' and g['ws'] == 0.0), key=lambda g: g['NDCG@10'])
    # reference on test: full rule, all weights 1, beam 50
    ref = S.metrics(S.beam(mt, *Tt['full'], 50)[:, :20], tt)
    res = {'valid_users': len(tv), 'test_users': len(tt), 'chosen_on_valid': best, 'test': {}}
    rows = [('reference: full, beam-50, w=1', ref)]
    for name, g in [('full, exact, w=1', {'rule': 'full', 'wb': 1, 'wc': 1, 'ws': 1}),
                    ('codebook, exact, w=1', {'rule': 'codebook', 'wb': 1, 'wc': 1, 'ws': 0}),
                    ('trie, exact, w=1', {'rule': 'trie', 'wb': 1, 'wc': 1, 'ws': 0})] + \
                   [(f'{k}, exact, tuned w', v) for k, v in best.items()]:
        per = S.metrics(S.exhaustive_rank(total(mt, Tt[g['rule']], g['wb'], g['wc'], g['ws']))[:, :20], tt)
        rows.append((name, per))
        # the same weights under beam-50
        sa, sb, sc, suf = Tt[g['rule']]
        bm = S.beam(mt, sa, g['wb'] * sb, g['wc'] * sc, g['ws'] * suf, 50)
        rows.append((name.replace('exact', 'beam-50'), S.metrics(bm[:, :20], tt)))
    lines = ['| test configuration (weights chosen on validation) | HR@1 | HR@5 | HR@10 | NDCG@10 | ΔHR@10 vs ref [95% CI] | ΔNDCG@10 vs ref [95% CI] |',
             '|---|---:|---:|---:|---:|---|---|']
    for name, per in rows:
        sm = S.summarize(per)
        dh = S.boot(per['HR@10'], ref['HR@10']); dn = S.boot(per['NDCG@10'], ref['NDCG@10'])
        res['test'][name] = {**sm, 'dHR10': dh, 'dNDCG10': dn}
        lines.append(f"| {name} | {sm['HR@1']:.4f} | {sm['HR@5']:.4f} | {sm['HR@10']:.4f} | {sm['NDCG@10']:.4f} | "
                     f"{dh[0]*100:+.2f} [{dh[1]*100:+.2f}, {dh[2]*100:+.2f}] | {dn[0]*100:+.2f} [{dn[1]*100:+.2f}, {dn[2]*100:+.2f}] |")
    table = '\n'.join(lines)
    print('chosen on validation:', json.dumps(best, indent=1))
    print(table)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        Path(a.out).with_suffix('.md').write_text('chosen on validation:\n\n```\n' + json.dumps(best, indent=1) + '\n```\n\n' + table + '\n')


if __name__ == '__main__':
    main()
