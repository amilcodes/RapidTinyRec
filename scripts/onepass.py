"""Can the second forward be removed? Offline study on exhaustive FP32 dumps.

usage: onepass.py VALID_DUMP TEST_DUMP [--out res.json]

Certified exact decoding needs forward 1 (prompt + all first codes -> s(a), s(b|a) for every
prefix) and forward 2 (top prefixes -> s(c|ab)). This script asks:
  1. How user-dependent is the third-code term?  Variance decomposition of s(c|ab) into an
     item-identity part (mean over validation users) and a user-specific residual.
  2. Does a user-independent third-code table (fit on the validation split) recover the quality
     of the per-user term?  If yes, recommendation needs ONE forward per request.
  3. How many prefixes does the certificate actually have to expand (|{p : U2(p) > s_K}|)?
All rankings are exhaustive (= certified exact). Bootstrap CIs are paired over test users.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as S


def prep(d):
    D, meta = S.load(d)
    sid_list = [tuple(x) for x in meta['sid_tok'].tolist()]
    pos_of = {s: i for i, s in enumerate(sid_list)}
    tgt = torch.tensor([pos_of.get(tuple(t), -1) for t in D['targets']])
    return D, meta, tgt


def u2_of(meta, sa, sb, wb=1.0):
    return (sa[..., None] + wb * sb).reshape(sa.shape[0], -1)[:, meta['sid_flat']]


def evaluate(total, tgt):
    return S.metrics(S.exhaustive_rank(total, 20), tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('valid'); ap.add_argument('test'); ap.add_argument('--out')
    ap.add_argument('--rule', default='codebook')
    ap.add_argument('--wc', type=float, default=0.5)
    a = ap.parse_args()
    Dv, meta, tv = prep(a.valid)
    Dt, _, tt = prep(a.test)
    res = {'valid_users': len(tv), 'test_users': len(tt), 'rule': a.rule, 'wc': a.wc}

    sav, sbv, scv, sufv = S.terms(Dv, meta, a.rule)
    sat, sbt, sct, suft = S.terms(Dt, meta, a.rule)
    # full-objective reference decode (MiniOneRec: full rule, beam 50)
    fa, fb, fc, fs = S.terms(Dt, meta, 'full')
    ref = S.metrics(S.beam(meta, fa, fb, fc, fs, 50)[:, :20], tt)
    res['reference_full_beam50'] = S.summarize(ref)

    # 1. variance decomposition of the third-code term (test users, validation item means)
    prior = scv.mean(0)                                   # [S] user-independent table
    resid = sct - prior[None]
    tot_var = sct.var().item()
    res['c_term'] = {
        'total_var': tot_var,
        'item_mean_var_explained': 1 - resid.var().item() / tot_var,
    }
    # restricted to items that matter: each user's top-50 under the per-user objective
    u2t = u2_of(meta, sat, sbt)
    tot_real = u2t + a.wc * sct + suft
    top50 = tot_real.topk(50, -1).indices
    sc50 = sct.gather(1, top50); pr50 = prior[top50]
    r50 = sc50 - pr50
    res['c_term']['top50_var'] = sc50.var().item()
    res['c_term']['top50_explained'] = 1 - r50.var().item() / sc50.var().item()
    # within-prefix ordering: does the prior order siblings like the user-specific term?
    pair = meta['sid_pair']
    agree, n = 0, 0
    for u in range(0, len(tt), 7):
        items = top50[u].tolist()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                si, sj = items[i], items[j]
                if pair[si] == pair[sj]:
                    n += 1
                    agree += int((sct[u, si] > sct[u, sj]) == (prior[si] > prior[sj]))
    res['c_term']['sibling_order_agreement'] = agree / max(n, 1)
    res['c_term']['sibling_pairs'] = n

    # 2. quality: per-user c (two forwards) vs static table (one forward) vs no c
    per = {}
    per['two_forward_real_c'] = evaluate(tot_real, tt)
    grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    u2v = u2_of(meta, sav, sbv)
    best, best_nd = None, -1
    val_rows = {}
    for w in grid:
        m = S.summarize(evaluate(u2v + w * prior[None] + sufv, tv))
        val_rows[w] = m
        if m['NDCG@10'] > best_nd:
            best, best_nd = w, m['NDCG@10']
    res['static_grid_valid'] = {str(k): v for k, v in val_rows.items()}
    res['static_w_chosen_on_valid'] = best
    per['one_forward_static_c'] = evaluate(u2t + best * prior[None] + suft, tt)
    per['one_forward_no_c'] = evaluate(u2t + 1e-4 * prior[None] + suft, tt)
    # oracle upper bound for a one-forward method: static w tuned on test
    res['summary'] = {k: S.summarize(v) for k, v in per.items()}
    res['bootstrap'] = {}
    for k, v in per.items():
        res['bootstrap'][f'{k}_vs_ref'] = {m: S.boot(v[m], ref[m]) for m in ('HR@1', 'HR@10', 'NDCG@10')}
        res['bootstrap'][f'{k}_vs_two_forward'] = {m: S.boot(v[m], per['two_forward_real_c'][m]) for m in ('HR@1', 'HR@10', 'NDCG@10')}

    # 3. certificate expansion sets: prefixes whose bound exceeds the K-th exact score
    P2 = int(pair.max()) + 1
    u2p = torch.full((len(tt), P2), float('-inf')).scatter_reduce(1, pair[None].expand_as(u2t), u2t, 'amax')
    srt = tot_real.sort(-1, descending=True).values
    cert = {}
    for K in (1, 10, 20, 50):
        need = (u2p > srt[:, K - 1:K]).sum(1).float()
        q = torch.quantile(need, torch.tensor([.5, .9, .99]))
        cert[K] = {'median': q[0].item(), 'p90': q[1].item(), 'p99': q[2].item(), 'max': need.max().item(),
                   'mean': need.mean().item()}
    res['prefixes_needed'] = cert
    # same with the static-c objective (what a speculative 2nd forward would need)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
