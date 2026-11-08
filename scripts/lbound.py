"""Two-sided bounds under trie normalization: guaranteed certification with a data-dependent expansion set.

With the third code log-softmax-normalized over the n valid children of (a, b), the best child has
log-prob >= -log n, so  U(p) - w_c log n_p  <=  max item score under p  <=  U(p),  U(p) = s(a) + w_b s(b|a).
After forward 1 (exact U for every prefix):
  tau_L = K-th largest of { exact single-child item scores } U { U(p) - w_c log n_p : p multi-child }
is a lower bound on the true K-th score tau*, and the set E = { multi-child p : U(p) >= tau_L } contains every
prefix that can hold a top-K item. Expanding E in forward 2 always certifies (the pool then holds >= K items
scoring >= tau_L > U of every unexpanded prefix), and |E| is known BEFORE forward 2. The minimal set is
N = { p : U(p) >= tau* } (any bound-based algorithm must expand it). We report |E| vs |N| vs K.

usage: lbound.py --release REL --late-entry .. --late-entry-a .. --out DIR
"""
import argparse, json, math, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactTrieEngine
from rapidtinyrec.speedrun import kernels
from inflight import template_of, load_bridge, batches, to_dev, forward1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--late-entry', default=None); ap.add_argument('--late-entry-a', default=None)
    ap.add_argument('--weights', default='1,0.75,0'); ap.add_argument('--norm', default='cbtrie')
    ap.add_argument('--users', type=int, default=0); ap.add_argument('--tag', default='office')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release); ids = data.prompts(a.release, tok, rows)
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, torch.float16, dev)
    Ks = [1, 5, 10, 20, 50, 100, 200, 500]
    eng = ExactTrieEngine(W, cat, n_prefix=400, n_out=500, norm=a.norm, ks=tuple(Ks))
    wb, wc, ws = (float(x) for x in a.weights.split(','))
    eng.set_weights(wb, wc, ws)
    eng.set_template(template_of(ids))
    if a.late_entry:
        eng.set_late_entry(*load_bridge(a.late_entry, W.D))
    if a.late_entry_a:
        eng.set_node_late(*load_bridge(a.late_entry_a, W.D))
    eng.fwd.attn = 'triton'; eng.fwd.tree_fast = True
    assert kernels.selfcheck(torch.float16)['pass'] and kernels.selfcheck_prompt(torch.float16)['pass']
    nchild = cat.v2.sum(-1).float()                                     # [pairs]
    nflat = torch.where(cat.v1.view(-1), nchild[cat.pair_table.view(-1)], torch.zeros(1, device=dev))
    logn = torch.log(nflat.clamp(min=1))
    multi, single = eng.multi_flat, eng.single_flat
    stats = {K: {'E': [], 'N': [], 'cert_engine': 0} for K in Ks}
    idx = list(range(len(ids))) if not a.users else list(range(a.users))
    nU = 0
    for chunk in batches(idx, ids, 8):
        t, ln = to_dev(chunk, ids, tok.eos_token_id, dev)
        u2, _, _, _, _ = forward1(eng, t, ln, [])
        r = eng.run(t, ln, eng.alloc(t.shape[0], t.shape[1]))
        lower = torch.cat([u2.masked_fill(~single[None], float('-inf')),
                           (u2 - wc * logn[None]).masked_fill(~multi[None], float('-inf'))], 1)
        srt = lower.sort(1, descending=True).values
        u2m = u2.masked_fill(~multi[None], float('-inf'))
        for j, K in enumerate(Ks):
            tauL = srt[:, K - 1]
            tau = r['scores'][:, K - 1]
            ok = r['certified'][:, j]
            stats[K]['E'] += (u2m >= tauL[:, None]).sum(1).tolist()
            stats[K]['N'] += (u2m >= tau[:, None]).sum(1).tolist()
            stats[K]['cert_engine'] += int(ok.sum())
            assert bool((tauL <= tau + 1e-3)[ok].all()), 'lower bound above the certified K-th score'
        nU += len(chunk)
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
    summ = {'users': nU, 'tag': a.tag, 'weights': [wb, wc, ws], 'norm': a.norm,
            'K': {K: {'E_mean': sum(s['E']) / nU, 'E_p50': q(s['E'], .5), 'E_p99': q(s['E'], .99), 'E_max': max(s['E']),
                      'N_mean': sum(s['N']) / nU, 'N_p99': q(s['N'], .99), 'N_max': max(s['N']),
                      'engine_M400_cert': s['cert_engine'] / nU} for K, s in stats.items()}}
    (out / f'lbound_{a.tag}.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({'stage': 'eval', **summ}), flush=True)


if __name__ == '__main__':
    main()
