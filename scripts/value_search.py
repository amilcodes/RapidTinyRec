"""Certified top-K under value-augmented objectives: score(i) = model score + lam * v[i].

Production rankers add item-level terms (popularity priors, bids, freshness) to the model's log-prob.
With beam search the term can only be applied after the beam has pruned (rerank), so high-value items
under low-probability prefixes are lost. Certified search stays exact: the bound of a prefix is
U2 + lam * max v over its items. We measure
  (1) popularity prior v = log(1 + train count): validation-tuned lam, test HR/NDCG, exact vs
      "top-50 by model score, then rerank" (an upper bound on any beam-50-then-rerank pipeline);
  (2) random log-normal "bids": search recall of the exact top-10 by the rerank pipeline, and
      certificate rates / cost (M) for the exact search.

usage: value_search.py --release TEST --valid-release VALID --train-csv CSV --late-entry .. --late-entry-a .. --out DIR
"""
import argparse, collections, csv, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactTrieEngine
from rapidtinyrec.speedrun import kernels
from inflight import template_of, load_bridge, batches, to_dev


def run_all(eng, ids, idx, eos, dev, B=16):
    """Returns {row: (sids list of tuples, scores list, certified list)}"""
    out = {}
    for chunk in batches(idx, ids, B):
        t, ln = to_dev(chunk, ids, eos, dev)
        with torch.no_grad():
            r = eng.run(t, ln, eng.alloc(t.shape[0], t.shape[1]))
        s = r['sids'].tolist(); sc = r['scores'].tolist(); c = r['certified'].tolist()
        for j, i in enumerate(chunk):
            out[i] = ([tuple(x) for x in s[j]], sc[j], c[j])
    return out


def metrics(res, targets, key=lambda r: r[0]):
    ms = [data.rank_metrics(key(res[i]), targets[i]) for i in res]
    return {k: sum(m[k] for m in ms) / len(ms) for k in ('HR@1', 'HR@5', 'HR@10', 'NDCG@10', 'HR@50')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--valid-release', required=True)
    ap.add_argument('--train-csv', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--late-entry', required=True); ap.add_argument('--late-entry-a', required=True)
    ap.add_argument('--weights', default='1,0.75,0'); ap.add_argument('--norm', default='cbtrie')
    ap.add_argument('--valid-users', type=int, default=3000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release); ids = data.prompts(a.release, tok, rows)
    targets = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    vrows = data.test_rows(a.valid_release); vids = data.prompts(a.valid_release, tok, vrows)
    vtargets = [data.sid_tokens(tok, r['item_sid']) for r in vrows]
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, torch.float16, dev)
    eos = tok.eos_token_id

    def engine(M, N=50):
        e = ExactTrieEngine(W, cat, n_prefix=M, n_out=N, norm=a.norm, ks=(1, 5, 10, 20, 50))
        e.set_weights(*(float(x) for x in a.weights.split(',')))
        e.set_template(template_of(ids))
        e.set_late_entry(*load_bridge(a.late_entry, W.D))
        e.set_node_late(*load_bridge(a.late_entry_a, W.D))
        e.fwd.attn = 'triton'; e.fwd.tree_fast = True
        return e
    assert kernels.selfcheck(torch.float16)['pass'] and kernels.selfcheck_prompt(torch.float16)['pass']
    eng = engine(32)
    S = eng.S
    sid_list = [tuple(x) for x in eng.sid_tok.tolist()]
    sid_index = {s: i for i, s in enumerate(sid_list)}
    # popularity prior from the training split (target items)
    cnt = collections.Counter()
    for r in csv.DictReader(open(a.train_csv)):
        s = data.sid_tokens(tok, r['item_sid'])
        if s in sid_index:
            cnt[s] += 1
    v_pop = torch.tensor([float(torch.log1p(torch.tensor(float(cnt[s])))) for s in sid_list])
    summ = {'items': S, 'train_items_seen': sum(1 for s in sid_list if cnt[s])}
    vidx = list(range(min(a.valid_users, len(vids))))
    tidx = list(range(len(ids)))
    # ---- (1) popularity prior: tune lam on validation (exact, certified), evaluate on test
    eng.set_item_bonus(None)
    base_v = run_all(eng, vids, vidx, eos, dev)
    base_t = run_all(eng, ids, tidx, eos, dev)
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    val = {}
    for lam in grid:
        if lam == 0:
            rv = base_v
        else:
            eng.set_item_bonus(lam * v_pop)
            rv = run_all(eng, vids, vidx, eos, dev)
        # rerank pipeline: model-score top-50, rerank by + lam v (upper bound on beam-50 + rerank)
        rr = {i: sorted(base_v[i][0], key=lambda s, i=i: -(base_v[i][1][base_v[i][0].index(s)] + lam * float(v_pop[sid_index[s]])))
              for i in base_v}
        val[lam] = {'exact': metrics(rv, vtargets), 'rerank50': metrics(rr, vtargets, key=lambda r: r),
                    'cert10': sum(rv[i][2][2] for i in rv) / len(rv)}
        print(json.dumps({'stage': 'eval', 'part': 'pop_valid', 'lam': lam, **val[lam]}), flush=True)
    best = max(grid, key=lambda l: val[l]['exact']['NDCG@10'])
    best_rr = max(grid, key=lambda l: val[l]['rerank50']['NDCG@10'])
    test = {}
    for lam, kind in ((best, 'exact'), (best_rr, 'rerank50')):
        if kind == 'exact':
            eng.set_item_bonus(lam * v_pop if lam else None)
            rt = run_all(eng, ids, tidx, eos, dev)
            test[kind] = {'lam': lam, **metrics(rt, targets), 'cert10': sum(rt[i][2][2] for i in rt) / len(rt)}
        else:
            rr = {i: sorted(base_t[i][0], key=lambda s, i=i: -(base_t[i][1][base_t[i][0].index(s)] + lam * float(v_pop[sid_index[s]])))
                  for i in base_t}
            test[kind] = {'lam': lam, **metrics(rr, targets, key=lambda r: r)}
    test['model_only'] = metrics(base_t, targets)
    summ['popularity'] = {'valid': val, 'test': test}
    print(json.dumps({'stage': 'eval', 'part': 'pop_test', **test}), flush=True)
    # ---- (2) random bids: search error of rerank vs exact, cost of certification
    g = torch.Generator().manual_seed(0)
    v_bid = torch.randn(S, generator=g)
    bids = {}
    sub = tidx[:2000]
    for lam in (0.25, 0.5, 1.0, 2.0):
        eng.set_item_bonus(lam * v_bid)
        rx = run_all(eng, ids, sub, eos, dev)
        rec = []
        for i in sub:
            ex10 = set(rx[i][0][:10])
            rr = sorted(base_t[i][0], key=lambda s, i=i: -(base_t[i][1][base_t[i][0].index(s)] + lam * float(v_bid[sid_index[s]])))
            rec.append(len(ex10 & set(rr[:10])) / 10)
        bids[lam] = {'rerank50_recall@10_of_exact': sum(rec) / len(rec), 'cert10_M32': sum(rx[i][2][2] for i in rx) / len(rx)}
        print(json.dumps({'stage': 'eval', 'part': 'bids', 'lam': lam, **bids[lam]}), flush=True)
    summ['bids'] = bids
    (out / 'value_search.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
