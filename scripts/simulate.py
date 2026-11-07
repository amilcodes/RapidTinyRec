"""Offline search x scoring-rule study from exhaustive score dumps.

usage: python scripts/simulate.py DUMP_DIR [--out results.json]

Scoring rules (per-token terms, all <= 0):
  full      lp(a)+lp(b|a)+lp(c|ab)+lp(\\n)+lp(EOS)   (MiniOneRec objective, full-vocab)
  nosuffix  lp(a)+lp(b|a)+lp(c|ab)                     (forced tokens dropped)
  codebook  each SID term renormalized over its level's codebook (FlashRec-style)
  trie      each SID term renormalized over valid trie children
  prefix    lp(a)+lp(b|a) (item ranked by its (a,b) prefix; ties by full c)
Search: exhaustive top-K; beam width w; prefix-filter top-N by U2 then rank by rule.
"""
import argparse, json, math, random, statistics
from pathlib import Path
import torch

KS = (1, 3, 5, 10, 20)


def load(d):
    parts = [torch.load(p, weights_only=False) for p in sorted(Path(d).glob('scores_*.pt'))]
    out = {k: torch.cat([p[k] for p in parts]) for k in ('la', 'cba', 'lb', 'cbb', 'lc', 'cbc', 'lnl', 'leos', 'rows')}
    meta = parts[0]['meta']
    out['targets'] = sum((p['meta']['targets'] for p in parts), [])
    order = out['rows'].argsort()
    for k in list(out):
        if k == 'targets':
            out[k] = [out[k][i] for i in order.tolist()]
        else:
            out[k] = out[k][order]
    return out, meta


def lse_groups(x, groups, n_groups):
    """logsumexp of x[..., j] within groups[j]; returns [..., n_groups]."""
    m = torch.full(x.shape[:-1] + (n_groups,), float('-inf'))
    m = m.scatter_reduce(-1, groups.expand_as(x), x, 'amax', include_self=True)
    ex = (x - m.gather(-1, groups.expand_as(x))).exp()
    s = torch.zeros(x.shape[:-1] + (n_groups,)).scatter_add(-1, groups.expand_as(x), ex)
    return m + s.log()


def terms(D, meta, rule):
    """Per-user step scores under a rule: sa [U,nA], sb [U,nA,W1], sc [U,S], suf [U,S]."""
    la, lb, lc = D['la'], D['lb'], D['lc']
    U, nA, W1 = lb.shape
    zero = torch.zeros_like(lc)
    if rule in ('full', 'nosuffix', 'prefix'):
        sa, sb, sc = la, lb, lc
    elif rule == 'codebook':
        sa = la - D['cba'][:, None]
        sb = lb - D['cbb'][..., None]
        sc = lc - D['cbc']
    elif rule == 'trie':
        sa = la - torch.logsumexp(la, -1, keepdim=True)
        sb = lb - torch.logsumexp(lb, -1, keepdim=True)
        P2 = int(meta['sid_pair'].max()) + 1
        g = lse_groups(lc, meta['sid_pair'][None], P2)
        sc = lc - g.gather(-1, meta['sid_pair'][None].expand_as(lc))
    else:
        raise ValueError(rule)
    suf = D['lnl'] + D['leos'] if rule == 'full' else zero
    if rule == 'prefix':
        sc = sc * 1e-6  # tie-break only
    return sa, sb, sc, suf


def item_scores(meta, sa, sb, sc, suf):
    u2 = (sa[..., None] + sb).reshape(sa.shape[0], -1)[:, meta['sid_flat']]
    return u2, u2 + sc + suf


def exhaustive_rank(total, K=50):
    return total.topk(K, -1).indices


def beam(meta, sa, sb, sc, suf, w):
    """Constrained beam search of width w (same semantics as the engine)."""
    U, nA, W1 = sb.shape
    sid_flat = meta['sid_flat']
    S = sc.shape[1]
    # children of each (a, b-slot) as SID indices
    flat_children = {}
    for s, f in enumerate(sid_flat.tolist()):
        flat_children.setdefault(f, []).append(s)
    W2 = max(len(v) for v in flat_children.values())
    ch = torch.full((nA * W1, W2), -1, dtype=torch.long)
    for f, v in flat_children.items():
        ch[f, :len(v)] = torch.tensor(v)
    s1, a = sa.topk(min(w, nA), -1)                                   # [U,w]
    c1 = s1[..., None] + sb.gather(1, a[..., None].expand(-1, -1, W1))  # [U,w,W1]
    c1 = c1.reshape(U, -1)
    s2, j = c1.topk(min(w, c1.shape[1]), -1)
    flat = a.gather(1, j // W1) * W1 + j % W1                         # [U,w2]
    kids = ch[flat]                                                    # [U,w2,W2]
    valid = kids >= 0
    kc = kids.clamp(min=0)
    c2 = (s2[..., None] + sc.gather(1, kc.reshape(U, -1)).reshape(kc.shape)).masked_fill(~valid, float('-inf'))
    s3, k = c2.reshape(U, -1).topk(min(w, c2.shape[1] * c2.shape[2]), -1)
    sids = kc.reshape(U, -1).gather(1, k)
    final = s3 + suf.gather(1, sids)
    final = final.masked_fill(torch.isinf(s3), float('-inf'))
    o = final.argsort(-1, descending=True)
    return sids.gather(1, o)


def prefix_filter(u2, total, N):
    sel = u2.topk(N, -1).indices
    o = total.gather(1, sel).argsort(-1, descending=True)
    return sel.gather(1, o)


def metrics(ranked, target_idx):
    """ranked [U,K] sid indices; target_idx [U] (-1 if target not in catalog)."""
    hit = ranked == target_idx[:, None]
    pos = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(target_idx, 10**6))
    per = {}
    for k in KS:
        h = (pos < k).float()
        per[f'HR@{k}'] = h
        per[f'NDCG@{k}'] = torch.where(pos < k, 1 / torch.log2(pos.float() + 2), torch.zeros_like(h))
    return per


def summarize(per):
    return {k: v.mean().item() for k, v in per.items()}


def boot(a, b, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = a - b
    idx = torch.randint(0, len(d), (n, len(d)), generator=g)
    bs = d[idx].mean(1).sort().values
    return d.mean().item(), bs[int(.025 * n)].item(), bs[int(.975 * n)].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D, meta = load(a.dump)
    sid_list = [tuple(x) for x in meta['sid_tok'].tolist()]
    pos_of = {s: i for i, s in enumerate(sid_list)}
    tgt = torch.tensor([pos_of.get(tuple(t), -1) for t in D['targets']])
    U = len(tgt)
    res = {'users': U}
    per_store = {}
    for rule in ('full', 'nosuffix', 'codebook', 'trie', 'prefix'):
        sa, sb, sc, suf = terms(D, meta, rule)
        u2, total = item_scores(meta, sa, sb, sc, suf)
        ex = exhaustive_rank(total)
        per = metrics(ex[:, :20], tgt)
        per_store[(rule, 'exhaustive')] = per
        res[f'{rule}/exhaustive'] = summarize(per)
        for w in (1, 5, 10, 20, 50, 100):
            bm = beam(meta, sa, sb, sc, suf, w)
            per = metrics(bm[:, :20], tgt)
            per_store[(rule, f'beam{w}')] = per
            r = summarize(per)
            k = min(10, bm.shape[1])
            r['search_recall@10'] = (torch.stack([torch.isin(bm[u, :k], ex[u, :10]).float().sum() / 10 for u in range(U)])).mean().item()
            r['top1_exact'] = (bm[:, 0] == ex[:, 0]).float().mean().item()
            res[f'{rule}/beam{w}'] = r
        # prefix-filter: rank by rule, admit top-N by the *full-vocab* (a,b) bound
        fa, fb, _, _ = terms(D, meta, 'full')
        u2_full, _ = item_scores(meta, fa, fb, sc, suf)
        for N in (8, 16, 32, 64, 128):
            pf = prefix_filter(u2_full, total, N)
            per = metrics(pf[:, :20], tgt)
            per_store[(rule, f'filter{N}')] = per
            res[f'{rule}/filter{N}'] = summarize(per)
    # lambda-weighted tail: lp(a)+lp(b|a) + lam*(lp(c|ab)+lp(\n)+lp(EOS)), exhaustive search,
    # lambda chosen by 2-fold cross-fitting over users (fit on one half, report on the other)
    fa, fb, fc, fs = terms(D, meta, 'full')
    u2_full, _ = item_scores(meta, fa, fb, fc * 0, fs * 0)
    lam_grid = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    lam_per = {}
    for lam in lam_grid:
        tot = u2_full + lam * (fc + fs)
        per = metrics(exhaustive_rank(tot)[:, :20], tgt)
        lam_per[lam] = per
        res[f'lambda{lam}/exhaustive'] = summarize(per)
        per_store[(f'lambda{lam}', 'exhaustive')] = per
    folds = torch.arange(U) % 2
    cf = {m: torch.zeros(U) for m in lam_per[1.0]}
    chosen = []
    for f in (0, 1):
        fit = folds == f
        best = max(lam_grid, key=lambda l: lam_per[l]['NDCG@10'][fit].mean().item())
        chosen.append(best)
        for m in cf:
            cf[m][~fit] = lam_per[best][m][~fit]
    per_store[('lambda-crossfit', 'exhaustive')] = cf
    res['lambda-crossfit/exhaustive'] = summarize(cf)
    res['lambda-crossfit/exhaustive']['chosen'] = chosen
    base = per_store[('full', 'beam50')]
    res['bootstrap_vs_full_beam50'] = {}
    for key, per in per_store.items():
        if key == ('full', 'beam50'):
            continue
        res['bootstrap_vs_full_beam50']['/'.join(key)] = {m: boot(per[m], base[m]) for m in ('HR@10', 'NDCG@10', 'HR@5')}
    lines = ['| rule/search | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | ΔHR@10 vs full/beam50 [95% CI] | search recall@10 | top1=exact |',
             '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for k, v in res.items():
        if not isinstance(v, dict) or 'HR@10' not in v:
            continue
        bt = res['bootstrap_vs_full_beam50'].get(k, {}).get('HR@10')
        bts = f'{bt[0]*100:+.2f} [{bt[1]*100:+.2f}, {bt[2]*100:+.2f}]' if bt else '—'
        lines.append(f"| {k} | {v['HR@1']:.4f} | {v['HR@5']:.4f} | {v['HR@10']:.4f} | {v['NDCG@5']:.4f} | {v['NDCG@10']:.4f} | {bts} | "
                     f"{v.get('search_recall@10', float('nan')):.3f} | {v.get('top1_exact', float('nan')):.3f} |")
    table = '\n'.join(lines)
    print(table)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        Path(a.out).with_suffix('.md').write_text(table + '\n')


if __name__ == '__main__':
    main()
