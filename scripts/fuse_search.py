"""Certified exact search with a collaborative value term: score = LLM score + beta * log q_SASRec(i | history).

beta (and optionally w_c) tuned on validation; test reports exact (certified) vs "LLM top-50 then rerank"
(an upper bound on any beam-50 + rerank pipeline) vs LLM only, with paired bootstrap CIs.

usage: fuse_search.py --release REL --valid-release VALID --sasrec DIR --late-entry .. --late-entry-a .. --out DIR
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


def run(eng, ids, idx, eos, dev, logq=None, beta=0.0, B=16):
    out = {}
    for chunk in batches(idx, ids, B):
        t, ln = to_dev(chunk, ids, eos, dev)
        eng.set_item_bonus(beta * logq[chunk].float() if beta else None)
        with torch.no_grad():
            r = eng.run(t, ln, eng.alloc(t.shape[0], t.shape[1]))
        s = r['sids'].tolist(); sc = r['scores'].tolist(); c = r['certified'].tolist()
        for j, i in enumerate(chunk):
            out[i] = ([tuple(x) for x in s[j]], sc[j], c[j])
    eng.set_item_bonus(None)
    return out


def per_user(lists, targets):
    return {i: data.rank_metrics(lists[i], targets[i]) for i in lists}


def agg(pu, keys=('HR@1', 'HR@5', 'HR@10', 'NDCG@10', 'HR@50')):
    return {k: sum(m[k] for m in pu.values()) / len(pu) for k in keys}


def boot(pa, pb, key, n=2000):
    ids = list(pa)
    d = [pa[i][key] - pb[i][key] for i in ids]
    rng = random.Random(0)
    m = sum(d) / len(d)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return [100 * m, 100 * bs[int(.025 * n)], 100 * bs[int(.975 * n)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--valid-release', required=True)
    ap.add_argument('--sasrec', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--late-entry', default=None); ap.add_argument('--late-entry-a', default=None)
    ap.add_argument('--weights', default='1,0.75,0'); ap.add_argument('--norm', default='cbtrie')
    ap.add_argument('--betas', default='0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0')
    ap.add_argument('--tag', default='office')
    ap.add_argument('--large-k', action='store_true', help='also: certified top-500 with the fused objective')
    ap.add_argument('--beta', type=float, default=None, help='skip validation; use this beta')
    ap.add_argument('--wc-grid', default=None, help='joint grid: third-code weights w_c (b weight 1), e.g. 0.5,0.75,1')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release); ids = data.prompts(a.release, tok, rows)
    tg = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    vrows = data.test_rows(a.valid_release); vids = data.prompts(a.valid_release, tok, vrows)
    vtg = [data.sid_tokens(tok, r['item_sid']) for r in vrows]
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, torch.float16, dev)
    eng = ExactTrieEngine(W, cat, n_prefix=32, n_out=50, norm=a.norm, ks=(1, 5, 10, 20, 50))
    eng.set_weights(*(float(x) for x in a.weights.split(',')))
    eng.set_template(template_of(ids))
    if a.late_entry:
        eng.set_late_entry(*load_bridge(a.late_entry, W.D))
    if a.late_entry_a:
        eng.set_node_late(*load_bridge(a.late_entry_a, W.D))
    eng.fwd.attn = 'triton'; eng.fwd.tree_fast = True
    assert kernels.selfcheck(torch.float16)['pass'] and kernels.selfcheck_prompt(torch.float16)['pass']
    sv = torch.load(Path(a.sasrec) / 'sasrec_valid.pt'); st = torch.load(Path(a.sasrec) / 'sasrec_test.pt')
    assert [tuple(x) for x in sv['sids']] == [tuple(x) for x in eng.sid_tok.tolist()]
    lq_v, lq_t = sv['logq'].to(dev), st['logq'].to(dev)
    lq_v = lq_v.clamp(min=-30.0); lq_t = lq_t.clamp(min=-30.0)          # items unseen by SASRec: finite floor
    sid_index = {tuple(s): i for i, s in enumerate(eng.sid_tok.tolist())}
    vidx, tidx = list(range(len(vids))), list(range(len(ids)))
    eos = tok.eos_token_id
    if a.large_k:
        assert a.beta is not None
        big = ExactTrieEngine(W, cat, n_prefix=256, n_out=500, norm=a.norm, ks=(10, 50, 100, 200, 500))
        big.set_weights(*(float(x) for x in a.weights.split(',')))
        big.tmpl, big.template_ids = eng.tmpl, eng.template_ids
        big.late, big.bridge, big.node_late_cfg = eng.late, eng.bridge, eng.node_late_cfg
        big.fwd = eng.fwd
        tidx = list(range(len(ids)))
        b0 = run(big, ids, tidx, eos, dev, B=8)
        b1 = run(big, ids, tidx, eos, dev, lq_t, a.beta, B=8)
        rr = {i: sorted(b0[i][0], key=lambda s, i=i: -(b0[i][1][b0[i][0].index(s)] + a.beta * float(lq_t[i, sid_index[s]])))
              for i in b0}

        def recall(lists, K):
            return sum(tg[i] in lists[i][:K] for i in lists) / len(lists)
        res = {}
        for K in (10, 50, 100, 200, 500):
            res[K] = {'llm': recall({i: b0[i][0] for i in b0}, K), 'rerank500': recall(rr, K),
                      'exact_fused': recall({i: b1[i][0] for i in b1}, K)}
        cert = {K: sum(b1[i][2][j] for i in b1) / len(b1) for j, K in enumerate((10, 50, 100, 200, 500))}
        print(json.dumps({'stage': 'eval', 'part': 'largeK', 'beta': a.beta, 'HR': res, 'cert_fused': cert}), flush=True)
        (out / f'fuse_largek_{a.tag}.json').write_text(json.dumps({'beta': a.beta, 'HR': res, 'cert_fused': cert}, indent=1))
        return
    wcs = [float(x) for x in a.wc_grid.split(',')] if a.wc_grid else [eng.level_w[2]]
    wb = eng.level_w[1]
    val, joint = {}, {}
    for wc in wcs:
        eng.set_weights(wb, wc, 0.0)
        base_v = run(eng, vids, vidx, eos, dev)
        for beta in [float(x) for x in a.betas.split(',')]:
            rv = base_v if beta == 0 else run(eng, vids, vidx, eos, dev, lq_v, beta)
            rr = {i: sorted(base_v[i][0], key=lambda s, i=i: -(base_v[i][1][base_v[i][0].index(s)] + beta * float(lq_v[i, sid_index[s]])))
                  for i in base_v}
            rec = {'exact': agg(per_user({i: rv[i][0] for i in rv}, vtg)), 'rerank50': agg(per_user(rr, vtg)),
                   'cert10': sum(rv[i][2][2] for i in rv) / len(rv)}
            joint[(wc, beta)] = rec
            if wc == wcs[0] or len(wcs) == 1:
                val[beta] = rec
            print(json.dumps({'stage': 'eval', 'part': 'valid', 'wc': wc, 'beta': beta, **rec}), flush=True)
    wc_ex, b_ex = max(joint, key=lambda k: joint[k]['exact']['NDCG@10'])
    wc_rr, b_rr = max(joint, key=lambda k: joint[k]['rerank50']['NDCG@10'])
    wc0 = max(wcs, key=lambda w: joint[(w, 0.0)]['exact']['NDCG@10']) if (wcs[0], 0.0) in joint else wcs[0]
    eng.set_weights(wb, wc0, 0.0)
    base_t = run(eng, ids, tidx, eos, dev)                        # LLM alone at its own best w_c
    eng.set_weights(wb, wc_rr, 0.0)
    base_rr = base_t if wc_rr == wc0 else run(eng, ids, tidx, eos, dev)
    eng.set_weights(wb, wc_ex, 0.0)
    rt = run(eng, ids, tidx, eos, dev, lq_t, b_ex)
    rr = {i: sorted(base_rr[i][0], key=lambda s, i=i: -(base_rr[i][1][base_rr[i][0].index(s)] + b_rr * float(lq_t[i, sid_index[s]])))
          for i in base_rr}
    pu_llm = per_user({i: base_t[i][0] for i in base_t}, tg)
    pu_ex = per_user({i: rt[i][0] for i in rt}, tg)
    pu_rr = per_user(rr, tg)
    test = {'beta_exact': b_ex, 'beta_rerank': b_rr, 'wc_exact': wc_ex, 'wc_rerank': wc_rr, 'wc_llm': wc0, 'llm': agg(pu_llm), 'exact_fused': agg(pu_ex), 'rerank50_fused': agg(pu_rr),
            'cert10_fused': sum(rt[i][2][2] for i in rt) / len(rt),
            'd_exact_vs_llm': {k: boot(pu_ex, pu_llm, k) for k in ('HR@1', 'HR@10', 'NDCG@10', 'HR@50')},
            'd_exact_vs_rerank': {k: boot(pu_ex, pu_rr, k) for k in ('HR@1', 'HR@10', 'NDCG@10', 'HR@50')}}
    print(json.dumps({'stage': 'eval', 'part': 'test', **test}), flush=True)
    (out / f'fuse_{a.tag}.json').write_text(json.dumps({'valid': val, 'joint': {f'{k[0]},{k[1]}': v for k, v in joint.items()},
                                                        'test': test}, indent=1))
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
