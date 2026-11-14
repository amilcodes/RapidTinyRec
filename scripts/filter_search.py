"""Certified exact top-K under item filters (inventory, region, policy) vs "LLM top-N, then filter".

A filter is an item term of 0 (eligible) or -inf (ineligible): the prefix bound becomes -inf when no
eligible item lies below the prefix, so certified search returns the exact top-K *eligible* items and
still certifies. Baseline: the LLM's certified top-50 / top-500 (unfiltered), filtered afterwards — the
best any beam-N-then-filter pipeline can do. Filters: random (per item) at several eligibility rates, and
a "category" filter (a random subset of first codes).

usage: filter_search.py --release REL --late-entry .. --late-entry-a .. --out DIR
"""
import argparse, json, random, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactTrieEngine
from rapidtinyrec.speedrun import kernels
from inflight import template_of, load_bridge, batches, to_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--late-entry', default=None); ap.add_argument('--late-entry-a', default=None)
    ap.add_argument('--weights', default='1,0.75,0'); ap.add_argument('--norm', default='cbtrie')
    ap.add_argument('--users', type=int, default=2000); ap.add_argument('--tag', default='office')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release); ids = data.prompts(a.release, tok, rows)
    tg = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, torch.float16, dev)
    assert kernels.selfcheck(torch.float16)['pass'] and kernels.selfcheck_prompt(torch.float16)['pass']

    def engine(M, N, ks):
        e = ExactTrieEngine(W, cat, n_prefix=M, n_out=N, norm=a.norm, ks=ks)
        e.set_weights(*(float(x) for x in a.weights.split(',')))
        e.set_template(template_of(ids))
        if a.late_entry:
            e.set_late_entry(*load_bridge(a.late_entry, W.D))
        if a.late_entry_a:
            e.set_node_late(*load_bridge(a.late_entry_a, W.D))
        e.fwd.attn = 'triton'; e.fwd.tree_fast = True
        return e
    big = engine(256, 500, (10, 50, 100, 500))
    S = big.S
    sid_list = [tuple(x) for x in big.sid_tok.tolist()]
    idx = list(range(min(a.users, len(ids))))
    eos = tok.eos_token_id

    def run(e, mask=None, B=8):
        res = {}
        for chunk in batches(idx, ids, B):
            t, ln = to_dev(chunk, ids, eos, dev)
            e.set_item_bonus(None if mask is None else torch.where(mask, 0.0, float('-inf')))
            with torch.no_grad():
                r = e.run(t, ln, e.alloc(t.shape[0], t.shape[1]))
            s = r['sids'].tolist(); sc = r['scores'].tolist(); c = r['certified'].tolist()
            for j, i in enumerate(chunk):
                res[i] = ([tuple(x) for x in s[j]], sc[j], c[j])
        e.set_item_bonus(None)
        return res
    base = run(big)                                   # unfiltered certified top-500
    g = torch.Generator().manual_seed(0)
    filters = {}
    for rate in (0.5, 0.2, 0.05):
        filters[f'random_{rate}'] = torch.rand(S, generator=g) < rate
    first = sorted({s[0] for s in sid_list})
    rng = random.Random(0)
    for frac in (0.25, 0.1):
        keep = set(rng.sample(first, max(1, int(frac * len(first)))))
        filters[f'category_{frac}'] = torch.tensor([s[0] in keep for s in sid_list])
    summ = {'users': len(idx), 'tag': a.tag}
    for M in (32, 64):
        eng = engine(M, 50, (1, 5, 10, 20, 50))
        for name, mask in filters.items():
            md = mask.to(dev)
            ex = run(eng, md, B=16)
            elig = {sid_list[k] for k in range(S) if bool(mask[k])}
            rec50, rec500, hit_ex, hit_50, hit_500 = [], [], 0, 0, 0
            for i in idx:
                ex10 = set(ex[i][0][:10])
                f50 = [s for s in base[i][0][:50] if s in elig][:10]
                f500 = [s for s in base[i][0] if s in elig][:10]
                rec50.append(len(ex10 & set(f50)) / 10); rec500.append(len(ex10 & set(f500)) / 10)
                hit_ex += tg[i] in ex[i][0][:10]; hit_50 += tg[i] in f50; hit_500 += tg[i] in f500
            n = len(idx)
            rec = {'eligible_frac': float(mask.float().mean()), 'M': M,
                   'cert10': sum(ex[i][2][2] for i in idx) / n,
                   'recall@10_of_exact__top50_then_filter': sum(rec50) / n,
                   'recall@10_of_exact__top500_then_filter': sum(rec500) / n,
                   'HR@10_exact': hit_ex / n, 'HR@10_top50_filter': hit_50 / n, 'HR@10_top500_filter': hit_500 / n,
                   'avg_items_left_top50': sum(len([s for s in base[i][0][:50] if s in elig]) for i in idx) / n}
            summ[f'{name}/M{M}'] = rec
            print(json.dumps({'stage': 'eval', 'filter': name, **rec}), flush=True)
    (out / f'filter_{a.tag}.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
