"""Late entry for the third-code forward: can prefix rows skip the lower layers?

Certified exact decoding runs forward 1 (prompt + every first code a as a tree node) and forward 2
(the M best prefixes (a, b) as one token each) to get s(c|ab). This script replaces forward 2's
lower layers: a prefix row enters at layer e with an input built from its parent node's residual
stream at layer e (already computed in forward 1) and the token b, then runs layers e..L-1 with
exact attention to the prompt and parent K/V. Variants for the layer-e input:
  parent  h_a(e)                                     (no b information; floor)
  swap    h_a(e) + E[b] - E[a]
  ridge   h_a(e) + [h_a(e), E[b], E[a], 1] @ R_e       (R_e: ridge fit on held-out prompts, no labels)
  oracle  h_b(e) (the true residual; sanity check, must equal the full forward)
  exit    the usual forward truncated to its first L-e layers (early exit, same cost)
Scoring: codebook-normalized s(a) + s(b|a) + 0.5 s(c|ab) (validation-tuned Office objective).
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactEngine3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--entries', default='14,18,20,22,24,26')
    ap.add_argument('--fit-users', type=int, default=600)
    ap.add_argument('--eval-users', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--n-prefix', type=int, default=64)
    ap.add_argument('--weights', default='1,0.5,0')
    ap.add_argument('--norm', default='codebook')
    ap.add_argument('--dtype', default='float16')
    ap.add_argument('--ridge', type=float, default=1e-2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    E_LIST = [int(x) for x in a.entries.split(',')]
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release)
    all_ids = data.prompts(a.release, tok, rows)
    targets = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, getattr(torch, a.dtype), dev)
    eng = ExactEngine3(W, cat, n_prefix=a.n_prefix, n_expand=50, suffix_mode='none', norm=a.norm, fuse_first=True)
    eng.set_weights(*(float(x) for x in a.weights.split(',')))
    lcp = list(all_ids[0])
    for ids_ in all_ids:
        k = 0
        while k < min(len(lcp), len(ids_)) and lcp[k] == ids_[k]:
            k += 1
        lcp = lcp[:k]
    eng.set_template(lcp)
    fwd, M = eng.fwd, eng.M
    nA = len(cat.first)
    E = W.embed
    D = W.D
    fit_ids = [i for i in range(len(rows)) if i % 5 == 4][:a.fit_users]
    eval_ids = [i for i in range(len(rows)) if i % 5 != 4][:a.eval_users]
    print(json.dumps({'stage': 'setup', 'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'mps', 'fit': len(fit_ids), 'eval': len(eval_ids),
                      'entries': E_LIST, 'template': len(lcp)}), flush=True)

    def stage(chunk):
        B = len(chunk)
        P = max(len(all_ids[i]) for i in chunk)
        P = (P + 7) // 8 * 8
        t = torch.full((B, P), tok.eos_token_id, dtype=torch.long)
        for j, i in enumerate(chunk):
            t[j, :len(all_ids[i])] = torch.tensor(all_ids[i])
        t = t.to(dev)
        lengths = torch.tensor([len(all_ids[i]) for i in chunk], device=dev)
        bufs = eng.alloc(B, P)
        T0 = eng._T0()
        capA = {e: None for e in E_LIST}
        hp, hn = fwd.prefill_tree(E[t[:, T0:]], E[cat.first][None].expand(B, nA, D), lengths,
                                  bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=eng.tmpl, capture=capA)
        bi = torch.arange(B, device=dev)
        la = eng._lp(hp[bi, lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        lb = eng._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * cat.W1)
        u2 = la.repeat_interleave(cat.W1, dim=1) + lb
        u2m, pf = u2.topk(M, -1)
        ai, col = pf // cat.W1, pf % cat.W1
        b_tok = cat.t1[ai, col]
        a_tok = cat.first[ai]
        pair = cat.pair_table[ai, col]
        bidx = bi[:, None]
        for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
            kp[:, :, :, 0] = kp2[:, bidx, ai, 0]
        L0 = lengths[:, None, None]
        pos = (L0 + 1).expand(B, M, 1)
        pmask = torch.zeros(B, P, device=dev)
        pmask = torch.where(torch.arange(P, device=dev)[None] < lengths[:, None], 0.0, torch.finfo(torch.float32).min)
        t2, v2 = cat.t2[pair], cat.v2[pair]

        def c_scores(x0, start=0, n_layers=None, capture=None):
            h = fwd.step(x0, pos, bufs['kc'], bufs['vc'], pmask, bufs['kp3'], bufs['vp3'], 1, lengths=lengths,
                         n_layers=n_layers, start=start, capture=capture)
            return eng._lp(h[:, :, 0], 2, t2, v2).masked_fill(~v2, float('-inf'))

        capB = {e: None for e in E_LIST}
        lc_full = c_scores(E[b_tok][:, :, None], capture=capB)
        hA = {e: capA[e][bidx, ai][:, :, None] for e in E_LIST}           # [B,M,1,D]
        return dict(B=B, chunk=chunk, u2m=u2m, ai=ai, b_tok=b_tok, a_tok=a_tok, t2=t2, v2=v2, lc_full=lc_full,
                    hA=hA, hB=capB, c_scores=c_scores)

    # ---- fit ridge bridges R_e on held-out prompts (self-supervised: no targets used)
    feats = 3 * D + 1
    XtX = {e: torch.zeros(feats, feats, device=dev, dtype=torch.float64 if dev.type == 'cuda' else torch.float32) for e in E_LIST}
    XtY = {e: torch.zeros(feats, D, device=dev, dtype=torch.float64 if dev.type == 'cuda' else torch.float32) for e in E_LIST}
    s0 = time.time()
    with torch.inference_mode():
        for k in range(0, len(fit_ids), a.batch):
            st = stage(fit_ids[k:k + a.batch])
            eb = E[st['b_tok']].float().reshape(-1, D); ea = E[st['a_tok']].float().reshape(-1, D)
            for e in E_LIST:
                ha = st['hA'][e].float().reshape(-1, D)
                hb = st['hB'][e].float().reshape(-1, D)
                X = torch.cat([ha, eb, ea, torch.ones_like(ha[:, :1])], 1)
                X = X.to(XtX[e].dtype)
                XtX[e] += X.T @ X
                XtY[e] += X.T @ (hb - ha).to(X.dtype)
        R = {}
        fit_stats = {}
        for e in E_LIST:
            lam = a.ridge * XtX[e].diagonal().mean()
            R[e] = torch.linalg.solve(XtX[e] + lam * torch.eye(feats, device=dev, dtype=torch.float64 if dev.type == 'cuda' else torch.float32), XtY[e]).float()
    print(json.dumps({'stage': 'fit', 's': time.time() - s0}), flush=True)

    # ---- evaluate
    names = ['full'] + [f'{v}@{e}' for e in E_LIST for v in ('parent', 'swap', 'ridge', 'exit')] + [f'oracle@{E_LIST[0]}']
    per = {n: [] for n in names}
    agree = {n: [] for n in names}
    resid = {e: [] for e in E_LIST}
    s0 = time.time()
    with torch.inference_mode():
        for k in range(0, len(eval_ids), a.batch):
            st = stage(eval_ids[k:k + a.batch])
            B = st['B']
            eb, ea = E[st['b_tok']][:, :, None].float(), E[st['a_tok']][:, :, None].float()
            lcs = {'full': st['lc_full']}
            for e in E_LIST:
                ha = st['hA'][e].float()
                dt = st['hA'][e].dtype
                lcs[f'parent@{e}'] = st['c_scores'](ha.to(dt), start=e)
                lcs[f'swap@{e}'] = st['c_scores']((ha + eb - ea).to(dt), start=e)
                X = torch.cat([ha, eb, ea, torch.ones_like(ha[..., :1])], -1)
                xr = ha + X @ R[e]
                lcs[f'ridge@{e}'] = st['c_scores'](xr.to(dt), start=e)
                hb = st['hB'][e].float()
                resid[e].append(((xr - hb).norm(dim=-1) / (hb - ha).norm(dim=-1).clamp_min(1e-6)).flatten().cpu())
                lcs[f'exit@{e}'] = st['c_scores'](E[st['b_tok']][:, :, None], n_layers=W.L - e)
            lcs[f'oracle@{E_LIST[0]}'] = st['c_scores'](st['hB'][E_LIST[0]], start=E_LIST[0])
            full_tot = (st['u2m'][..., None] + st['lc_full']).view(B, -1)
            full_top = full_tot.topk(20, -1).indices
            # children valid count per prefix (>=2 means c-order matters)
            multi = st['v2'].sum(-1) >= 2
            for n, lc in lcs.items():
                tot = (st['u2m'][..., None] + lc).view(B, -1)
                top = tot.topk(20, -1).indices
                row, kk = top // cat.W2, top % cat.W2
                sid = torch.stack([st['a_tok'].gather(1, row), st['b_tok'].gather(1, row), st['t2'].view(B, -1).gather(1, top)], -1).cpu().tolist()
                for j, i in enumerate(st['chunk']):
                    per[n].append(data.rank_metrics([tuple(x) for x in sid[j]], targets[i], ks=(1, 5, 10, 20)))
                same1 = (top[:, 0] == full_top[:, 0]).float()
                ov10 = (top[:, :10, None] == full_top[:, None, :10]).any(-1).float().mean(-1)
                cm = (lc.argmax(-1) == st['lc_full'].argmax(-1)) | ~multi
                agree[n].append(torch.stack([same1, ov10, cm.float().sum(-1) / M], -1).cpu())
    res = {'setup': {'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'mps', 'eval_users': len(eval_ids), 'fit_users': len(fit_ids),
                     'M': M, 'weights': a.weights, 'norm': a.norm, 'dtype': a.dtype, 'eval_s': time.time() - s0}}
    for n in names:
        m = {key: statistics.mean(p[key] for p in per[n]) for key in per[n][0]}
        g = torch.cat(agree[n]).mean(0).tolist()
        res[n] = {**m, 'top1_same_as_full': g[0], 'top10_overlap_full': g[1], 'c_argmax_agree': g[2]}
        if n != 'full':
            d = [p['NDCG@10'] - q['NDCG@10'] for p, q in zip(per[n], per['full'])]
            h1 = [p['HR@1'] - q['HR@1'] for p, q in zip(per[n], per['full'])]
            res[n]['dNDCG@10_vs_full'] = statistics.mean(d)
            res[n]['dHR@1_vs_full'] = statistics.mean(h1)
    res['ridge_rel_residual'] = {e: torch.cat(resid[e]).median().item() for e in E_LIST}
    print(json.dumps({'stage': 'late_entry', **res}), flush=True)
    (out / 'late_entry.json').write_text(json.dumps(res, indent=1))
    torch.save({e: R[e].half().cpu() for e in E_LIST}, out / 'ridge_bridges.pt')


if __name__ == '__main__':
    main()
