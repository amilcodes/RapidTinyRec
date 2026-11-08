"""In-flight expansion analysis: can prefix rows be chosen *inside* forward 1?

The deployed engine runs forward 1 (prompt + first-level rows, late-entered at 18) to the top, picks the
M best multi-child prefixes by their exact bound U2 = s(a) + s(b|a), then runs forward 2 (prefix rows,
layers 22..27). If the prefix set could be chosen at layer 22 from an *estimate* of U2, the prefix rows
could join forward 1 there and the whole certified search would take one forward. The certificate is
still checked with exact quantities at the end (exact U2 of every prefix comes out of the same forward),
so a bad estimate only costs a fallback, never correctness.

This script fits a tuned lens (ridge, h_e -> final-normed state) on validation prompts and measures, on
test users, how often the top-M' prefixes by the lens estimate contain every prefix the certificate needs
(all multi-child prefixes with exact U2 >= the K-th final score). It also measures the same for the first
level at layer 18 (pruning first-code rows for throughput: an item under a is bounded by s(a)).

usage: inflight.py --train-release VALID --release TEST --late-entry 22:c.pt --late-entry-a 18:b.pt --out DIR
"""
import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactTrieEngine
from rapidtinyrec.speedrun.bridge import Bridge
from rapidtinyrec.speedrun import kernels


def template_of(all_ids):
    lcp = list(all_ids[0])
    for ids_ in all_ids:
        k = 0
        while k < min(len(lcp), len(ids_)) and lcp[k] == ids_[k]:
            k += 1
        lcp = lcp[:k]
    return lcp


def load_bridge(spec, D):
    e, p = spec.split(':', 1)
    sd = torch.load(p, map_location='cpu')
    br = Bridge(D, sd['inp.weight'].shape[0]); br.load_state_dict(sd)
    return int(e), br


def batches(idx, ids, B):
    idx = sorted(idx, key=lambda i: len(ids[i]))
    for s in range(0, len(idx), B):
        yield idx[s:s + B]


def to_dev(chunk, ids, eos, dev):
    P = (max(len(ids[i]) for i in chunk) + 7) // 8 * 8
    t = torch.full((len(chunk), P), eos, dtype=torch.long)
    for j, i in enumerate(chunk):
        t[j, :len(ids[i])] = torch.tensor(ids[i])
    return t.to(dev), torch.tensor([len(ids[i]) for i in chunk], device=dev)


@torch.no_grad()
def forward1(eng, t, lengths, layers):
    """Forward 1 as deployed; returns exact u2 [B,nA*W1], la [B,nA], and states entering `layers`
    for the last prompt token [B,D] and the first-level rows [B,nA,D] (None if not yet entered)."""
    W, cat, fwd = eng.W, eng.cat, eng.fwd
    B, P = t.shape
    nA = len(cat.first)
    T0 = eng._T0()
    bufs = eng.alloc(B, P)
    cap = {l: None for l in layers}
    capP = {l: None for l in layers}
    hp, hn = fwd.prefill_tree(W.embed[t[:, T0:]], W.embed[cat.first][None].expand(B, nA, W.D), lengths,
                              bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=eng.tmpl, capture=cap,
                              capture_prompt=capP, node_late=eng._node_late(t, lengths))
    bi = torch.arange(B, device=t.device)
    last = lengths - 1 - T0
    la = eng._lp(hp[bi, last], 0, cat.first[None].expand(B, -1), None)
    lb = eng._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
    lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * cat.W1)
    u2 = la.repeat_interleave(cat.W1, dim=1) + lb
    st = {l: (capP[l][bi, last], cap[l]) for l in layers}
    return u2, la, st, hp[bi, last], hn


def lens_scores(eng, lens, hP, hN):
    """Estimated la, lb from layer-e states via a ridge lens (or logit lens if lens is None)."""
    W, cat = eng.W, eng.cat
    B, nA = hP.shape[0], len(cat.first)

    def proj(h):
        h = h.float()
        if lens is None:
            return eng.fwd.final(h.to(W.dtype))
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (torch.cat([hn, torch.ones_like(hn[..., :1])], -1) @ lens).to(W.dtype)
    la = eng._lp(proj(hP), 0, cat.first[None].expand(B, -1), None)
    lb = None
    if hN is not None:
        lb = eng._lp(proj(hN), 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * cat.W1)
    return la, lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-release', required=True); ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--late-entry', required=True); ap.add_argument('--late-entry-a', required=True)
    ap.add_argument('--weights', default='1,0.75,0'); ap.add_argument('--norm', default='cbtrie')
    ap.add_argument('--train-users', type=int, default=4000); ap.add_argument('--test-users', type=int, default=0)
    ap.add_argument('--ridge', type=float, default=10.0); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--layers', default='18,20,22,24')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    layers = [int(x) for x in a.layers.split(',')]
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release); ids = data.prompts(a.release, tok, rows)
    vrows = data.test_rows(a.train_release); vids = data.prompts(a.train_release, tok, vrows)
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, torch.float16, dev)
    eng = ExactTrieEngine(W, cat, n_prefix=96, n_out=50, norm=a.norm, ks=(1, 5, 10, 20, 50))
    eng.set_weights(*(float(x) for x in a.weights.split(',')))
    eng.set_template(template_of(ids))
    assert template_of(vids)[:len(eng.template_ids)] == eng.template_ids
    eng.set_late_entry(*load_bridge(a.late_entry, W.D))
    eng.set_node_late(*load_bridge(a.late_entry_a, W.D))
    eng.fwd.attn = 'triton'; eng.fwd.tree_fast = True
    assert kernels.selfcheck(torch.float16)['pass'] and kernels.selfcheck_prompt(torch.float16)['pass']
    eos = tok.eos_token_id
    e_a = eng.node_late_cfg[0]
    # ---- fit ridge lenses on validation: features = rms-normed h_l (+1), target = final-normed state
    t0 = time.time()
    D = W.D
    XtX = {l: torch.zeros(D + 1, D + 1, device=dev, dtype=torch.float64) for l in layers}
    XtY = {l: torch.zeros(D + 1, D, device=dev, dtype=torch.float64) for l in layers}
    vidx = list(range(min(a.train_users, len(vids))))
    for chunk in batches(vidx, vids, a.batch):
        t, ln = to_dev(chunk, vids, eos, dev)
        _, _, st, hpl, hn = forward1(eng, t, ln, layers)
        for l in layers:
            hP, hN = st[l]
            X = [hP]; Y = [hpl]
            if hN is not None:
                X.append(hN.reshape(-1, D)); Y.append(hn.reshape(-1, D))
            X = torch.cat(X).double(); Y = torch.cat(Y).double()
            X = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + 1e-6)
            X = torch.cat([X, torch.ones_like(X[:, :1])], 1)
            XtX[l] += X.T @ X; XtY[l] += X.T @ Y
    lenses = {}
    for l in layers:
        n = XtX[l][-1, -1]
        reg = a.ridge * n / 1000 * torch.eye(D + 1, device=dev, dtype=torch.float64)
        reg[-1, -1] = 0
        lenses[l] = torch.linalg.solve(XtX[l] + reg, XtY[l]).float()
        torch.save(lenses[l].cpu(), out / f'lens_l{l}.pt')
    print(json.dumps({'stage': 'lens_fit', 's': time.time() - t0, 'users': len(vidx)}), flush=True)
    # ---- evaluate on test users
    Ms = [8, 12, 16, 24, 32, 48, 64]
    As = [8, 12, 16, 24, 32, 48]
    Ks = [10, 50]
    res = {f'{kind}/l{l}/K{K}': {M: 0 for M in Ms} for kind in ('ridge', 'logit') for l in layers for K in Ks}
    resA = {f'{kind}/l{l}/K{K}': {A: 0 for A in As} for kind in ('ridge', 'logit') for l in layers for K in Ks}
    need_sizes = {K: [] for K in Ks}
    needA_sizes = {K: [] for K in Ks}
    cert_ok = {K: 0 for K in Ks}
    nU = 0
    tidx = list(range(len(ids))) if not a.test_users else list(range(a.test_users))
    multi = eng.multi_flat[None]
    nA = len(cat.first)
    for chunk in batches(tidx, ids, a.batch):
        t, ln = to_dev(chunk, ids, eos, dev)
        B = t.shape[0]
        u2, la, st, _, _ = forward1(eng, t, ln, layers)
        r = eng.run(t, ln, eng.alloc(B, t.shape[1]))
        u2m = u2.masked_fill(~multi, float('-inf'))
        for K in Ks:
            kth = r['scores'][:, K - 1]
            ok = r['certified'][:, eng.ks.index(K)]
            cert_ok[K] += int(ok.sum())
            need = (u2m >= kth[:, None])                              # [B, nA*W1]
            needA = (la >= kth[:, None])                               # first codes with any possible top-K item
            need_sizes[K] += need.sum(1).tolist(); needA_sizes[K] += needA.sum(1).tolist()
            for l in layers:
                hP, hN = st[l]
                for kind, lens in (('ridge', lenses[l]), ('logit', None)):
                    la_h, lb_h = lens_scores(eng, lens, hP, hN)
                    if lb_h is not None:
                        # second-code rows exist at this layer: estimate U2 with lens la_h + lens lb_h
                        est = (la_h.repeat_interleave(cat.W1, 1) + lb_h).masked_fill(~multi, float('-inf'))
                        order = est.argsort(1, descending=True)
                        rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(order.shape[1], device=dev).expand_as(order))
                        worst = torch.where(need, rank, torch.full_like(rank, -1)).max(1).values  # worst rank among needed
                        for M in Ms:
                            res[f'{kind}/l{l}/K{K}'][M] += int((worst < M).sum())
                    oa = la_h.argsort(1, descending=True)
                    ra = torch.empty_like(oa); ra.scatter_(1, oa, torch.arange(nA, device=dev).expand_as(oa))
                    worstA = torch.where(needA, ra, torch.full_like(ra, -1)).max(1).values
                    for A in As:
                        resA[f'{kind}/l{l}/K{K}'][A] += int((worstA < A).sum())
        nU += B
    import statistics as S
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
    summ = {'users': nU, 'certified_M96': {K: cert_ok[K] / nU for K in Ks},
            'need_prefixes': {K: {'mean': S.mean(need_sizes[K]), 'p50': q(need_sizes[K], .5), 'p90': q(need_sizes[K], .9),
                                  'p99': q(need_sizes[K], .99), 'max': max(need_sizes[K])} for K in Ks},
            'need_first_codes': {K: {'mean': S.mean(needA_sizes[K]), 'p50': q(needA_sizes[K], .5), 'p90': q(needA_sizes[K], .9),
                                     'p99': q(needA_sizes[K], .99), 'max': max(needA_sizes[K])} for K in Ks},
            'prefix_recall': {k: {M: c / nU for M, c in v.items()} for k, v in res.items() if any(v.values())},
            'first_code_recall': {k: {A: c / nU for A, c in v.items()} for k, v in resA.items()},
            'first_level_entry': e_a}
    (out / 'inflight.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({'stage': 'inflight', **summ}), flush=True)


if __name__ == '__main__':
    main()
