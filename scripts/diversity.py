"""Exact top-K under diversity caps (at most `cap` items per category) from certified large-K lists (CPU).

A per-category cap is a partition matroid, so the exact capped top-K is the greedy pass over the global score
order. If a request's certified top-K' list (K' = 500) already yields K capped items, that list is exact for the
capped objective too (the greedy pass never needs an item below the K'-th). Baseline: top-50 then diversify, the
best any beam-50 + rerank pipeline can do (beam-50 returns at most the exact top-50). Categories: first SID code
(coarse) or the (a, b) prefix (fine).

usage: diversity.py --release REL --lists RUN/latency_users.json [...] --out FILE.json
"""
import argparse, csv, json, re
from pathlib import Path


def greedy(lst, key, cap, K):
    used, out = {}, []
    for s in lst:
        g = key(s)
        if used.get(g, 0) < cap:
            used[g] = used.get(g, 0) + 1
            out.append(s)
            if len(out) == K:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--lists', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='Office_Products'); ap.add_argument('--ckpt', default='Office_ckpt')
    a = ap.parse_args()
    added = json.load(open(Path(a.release) / a.ckpt / 'added_tokens.json'))
    tid = lambda s: tuple(added[t] for t in re.findall(r'<[a-z]_\d+>', s))
    rows = list(csv.DictReader(open(Path(a.release) / f'Amazon/test/{a.dataset}_5_2016-10-2018-11.csv')))
    K = 10
    res = {}
    for path, name in zip(a.lists, a.names):
        users = json.load(open(path))
        for gname, key in (('first_code', lambda s: s[0]), ('prefix_ab', lambda s: (s[0], s[1]))):
            for cap in (1, 2, 3):
                n = len(users)
                exact_ok = rec = hr_ex = hr_50 = full50 = 0
                distinct50 = 0
                for u in users:
                    lst = [tuple(x) for x in u['sids']]
                    tgt = tid(rows[u['row']]['item_sid'])
                    ex = greedy(lst, key, cap, K)
                    b50 = greedy(lst[:50], key, cap, K)
                    exact_ok += len(ex) == K
                    rec += len(set(ex) & set(b50)) / K
                    hr_ex += tgt in ex
                    hr_50 += tgt in b50
                    full50 += len(b50) == K
                    distinct50 += len({key(s) for s in lst[:50]})
                r = {'users': n, 'exact_within_500': exact_ok / n, 'top50_then_diversify_recall_of_exact': rec / n,
                     'top50_then_diversify_returns_K': full50 / n, 'HR@10_exact_capped': hr_ex / n,
                     'HR@10_top50_then_diversify': hr_50 / n, 'distinct_categories_in_top50': distinct50 / n}
                res[f'{name}/{gname}/cap{cap}'] = r
                print(name, gname, cap, json.dumps({k: round(v, 4) for k, v in r.items()}))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
