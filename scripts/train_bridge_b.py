"""Late entry for the first-level trie nodes too: a-node rows join forward 1 at layer e.

usage: train_bridge_b.py --train-release VALID_RELEASE --release TEST_RELEASE --out DIR --entry 22
                         --bridge-c BRIDGE_C.pt

In forward 1 the prompt rows are unaffected by the trie nodes (causal mask), so s(a) stays exact.
Here the 88 a-node rows skip layers < e: their layer-e input is bridge_b(h_e(last prompt token),
E[a], E[last token]); they run layers e..L-1 with exact attention to the prompt K/V. The bridge is
distilled from the full-depth s(b|a) distributions (codebook softmax over b codes) on validation
prompts. Evaluation on all test users (codebook, w_c = 0.5):
  full        both forwards at full depth (reference engine)
  b_late      late a-nodes, exact third code (isolates the s(b|a) approximation)
  b_late+c    late a-nodes and late prefix rows (c bridge applied to the late a-node state)
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_bridge import build
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.bridge import Bridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-release', required=True); ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--entry', type=int, default=22)
    ap.add_argument('--bridge-c', required=True)
    ap.add_argument('--train-users', type=int, default=3000); ap.add_argument('--ridge-users', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--hidden', type=int, default=2048)
    ap.add_argument('--weights', default='1,0.5,0')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.manual_seed(0); random.seed(0)
    wts = tuple(float(x) for x in a.weights.split(','))
    e = a.entry

    def f1(ctx, chunk, grad_bridge=None, functional=True):
        """Full forward 1 (teacher) and optionally the late a-node rows (student)."""
        tok, ids, cat, W, eng = ctx
        fwd = eng.fwd
        B = len(chunk)
        P = (max(len(ids[i]) for i in chunk) + 7) // 8 * 8
        t = torch.full((B, P), tok.eos_token_id, dtype=torch.long)
        for j, i in enumerate(chunk):
            t[j, :len(ids[i])] = torch.tensor(ids[i])
        t = t.to(dev)
        lengths = torch.tensor([len(ids[i]) for i in chunk], device=dev)
        bufs = eng.alloc(B, P)
        T0 = eng._T0()
        nA, D = len(cat.first), W.D
        capN, capP = {e: None}, {e: None}
        bi = torch.arange(B, device=dev)
        with torch.no_grad():
            hp, hn = fwd.prefill_tree(W.embed[t[:, T0:]], W.embed[cat.first][None].expand(B, nA, D), lengths,
                                      bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=eng.tmpl,
                                      capture=capN, capture_prompt=capP)
            la = eng._lp(hp[bi, lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        hlast = capP[e][bi, lengths - 1 - T0][:, None, None].expand(B, nA, 1, D)
        elast = W.embed[t[bi, lengths - 1]][:, None, None].expand(B, nA, 1, D)
        ea = W.embed[cat.first][None, :, None].expand(B, nA, 1, D)
        pmask = torch.where(torch.arange(P, device=dev)[None] < lengths[:, None], 0.0, torch.finfo(torch.float32).min)
        st = dict(B=B, chunk=chunk, t=t, lengths=lengths, bufs=bufs, la=la, hn=hn, hA=capN[e], pmask=pmask)
        if grad_bridge is not None:
            x0 = grad_bridge(hlast, ea, elast)
            st['x0b'] = x0
            h = fwd.step(x0.to(W.dtype), lengths[:, None, None].expand(B, nA, 1), bufs['kc'], bufs['vc'], pmask,
                         bufs['kp2'], bufs['vp2'], 0, lengths=lengths, start=e, functional=functional)
            st['hn_late'] = h[:, :, 0]                  # step() returns final-normed states
        st['feat'] = (hlast, ea, elast)
        return st

    def b_logp(eng, cat, h):
        b_idx = torch.nonzero(cat.level_mask[1]).flatten()
        return F.linear(h, eng.sid_head[b_idx]).float().log_softmax(-1)

    # ---------------- train on validation prompts (bf16)
    tok, rows, ids, _, cat, W, eng = build(a.train_release, torch.bfloat16, dev, wts)
    ctx = (tok, ids, cat, W, eng)
    users = list(range(len(rows))); random.shuffle(users)
    held, train = users[:200], users[200:200 + a.train_users]
    D = W.D
    br = Bridge(D, a.hidden).to(dev)
    br.scale.fill_(1.0 / W.embed[cat.sid_ids].float().pow(2).mean().sqrt().item())
    feats = 3 * D + 1
    XtX = torch.zeros(feats, feats, device=dev, dtype=torch.float64)
    XtY = torch.zeros(feats, D, device=dev, dtype=torch.float64)
    for k in range(0, a.ridge_users, a.batch):
        st = f1(ctx, train[k:k + a.batch])
        hl, ea, el = st['feat']
        X = torch.cat([hl.float(), ea.float(), el.float(), torch.ones_like(hl[..., :1]).float()], -1).reshape(-1, feats).double()
        XtX += X.T @ X
        XtY += X.T @ (st['hA'][:, :, None].float() - hl.float()).reshape(-1, D).double()
    br.R.copy_(torch.linalg.solve(XtX + 1e-2 * XtX.diagonal().mean() * torch.eye(feats, device=dev, dtype=torch.float64), XtY).float())
    del XtX, XtY

    def held_kl():
        tot, n = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(held), a.batch):
                st = f1(ctx, held[k:k + a.batch], grad_bridge=br)
                tgt = b_logp(eng, cat, st['hn'])
                lp = b_logp(eng, cat, st['hn_late'])
                tot += F.kl_div(lp, tgt, log_target=True, reduction='sum').item(); n += lp.shape[0] * lp.shape[1]
        return tot / n

    log = [{'epoch': 0, 'held_kl': held_kl()}]
    print(json.dumps({'stage': 'train_b', **log[-1]}), flush=True)
    opt = torch.optim.AdamW([p for p in br.parameters() if p.requires_grad], lr=a.lr)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        random.shuffle(train); run = []; t0 = time.time()
        for k in range(0, len(train), a.batch):
            st = f1(ctx, train[k:k + a.batch], grad_bridge=br)
            tgt = b_logp(eng, cat, st['hn'])
            lp = b_logp(eng, cat, st['hn_late'])
            loss = F.kl_div(lp, tgt, log_target=True, reduction='batchmean') / lp.shape[1]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(br.parameters(), 1.0); opt.step(); sched.step()
            run.append(loss.item())
        log.append({'epoch': ep + 1, 'train_kl': statistics.mean(run), 'held_kl': held_kl(), 's': time.time() - t0})
        print(json.dumps({'stage': 'train_b', **log[-1]}), flush=True)
    torch.save(br.state_dict(), out / f'bridge_b_e{e}.pt')
    del W, eng, ctx
    torch.cuda.empty_cache()

    # ---------------- evaluate on test (fp16)
    tok, rows, ids, targets, cat, W, eng = build(a.release, torch.float16, dev, wts)
    ctx = (tok, ids, cat, W, eng)
    sd = torch.load(a.bridge_c, map_location='cpu')
    brc = Bridge(W.D, sd['inp.weight'].shape[0]).to(dev); brc.load_state_dict(sd); brc.eval(); br.eval()
    nA, M, W1, W2 = len(cat.first), eng.M, cat.W1, cat.W2
    order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    per = {n: {} for n in ('full', 'b_late', 'b_late+c')}
    agree_b = []
    with torch.no_grad():
        for k in range(0, len(order), a.batch):
            chunk = order[k:k + a.batch]
            st = f1(ctx, chunk, grad_bridge=br, functional=True)
            B, bufs, lengths, pmask = st['B'], st['bufs'], st['lengths'], st['pmask']
            bi = torch.arange(B, device=dev)[:, None]
            L0 = lengths[:, None, None]
            hn_late = st['hn_late']                     # step() already applies the final norm
            results = {}
            for name, hn_ in (('full', st['hn']), ('b_late', hn_late)):
                lb = eng._lp(hn_, 1, cat.t1[None].expand(B, nA, W1), cat.v1[None].expand(B, nA, W1))
                lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * W1)
                u2m, pf = (st['la'].repeat_interleave(W1, dim=1) + lb).topk(M, -1)
                ai, col = pf // W1, pf % W1
                b_tok, pair = cat.t1[ai, col], cat.pair_table[ai, col]
                for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
                    kp[:, :, :, 0] = kp2[:, bi, ai, 0]
                h = eng.fwd.step(W.embed[b_tok][:, :, None], (L0 + 1).expand(B, M, 1), bufs['kc'], bufs['vc'], pmask,
                                 bufs['kp3'], bufs['vp3'], 1, lengths=lengths)
                results[name] = (u2m, ai, b_tok, pair, h[:, :, 0])
            # b_late + c late: rewrite a-node K/V at layers >= e with the late rows, then bridged prefix rows
            eng.fwd.step(st['x0b'].to(W.dtype), lengths[:, None, None].expand(B, nA, 1), bufs['kc'], bufs['vc'], pmask,
                         bufs['kp2'], bufs['vp2'], 0, lengths=lengths, start=e)
            u2m, ai, b_tok, pair, _ = results['b_late']
            for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
                kp[:, :, :, 0] = kp2[:, bi, ai, 0]
            ha = st['x0b'][bi, ai].float()                      # late a-node input at layer e = its residual there
            x0c = brc(ha, W.embed[b_tok][:, :, None], W.embed[cat.first[ai]][:, :, None])
            h = eng.fwd.step(x0c.to(W.dtype), (L0 + 1).expand(B, M, 1), bufs['kc'], bufs['vc'], pmask,
                             bufs['kp3'], bufs['vp3'], 1, lengths=lengths, start=e)
            results['b_late+c'] = (u2m, ai, b_tok, pair, h[:, :, 0])
            fa = results['full'][1][:, :10]; la_ = results['b_late'][1][:, :10]
            for name, (u2m, ai, b_tok, pair, hc) in results.items():
                t2, v2 = cat.t2[pair], cat.v2[pair]
                lc = eng._lp(hc, 2, t2, v2).masked_fill(~v2, float('-inf'))
                top = (u2m[..., None] + lc).view(B, -1).topk(20, -1).indices
                row = top // W2
                sid = torch.stack([cat.first[ai].gather(1, row), b_tok.gather(1, row), t2.view(B, -1).gather(1, top)], -1).cpu().tolist()
                for j, i in enumerate(chunk):
                    per[name][i] = data.rank_metrics([tuple(x) for x in sid[j]], targets[i], ks=(1, 5, 10, 20))
            # top-10 prefix overlap (b_late vs full)
            pf_full = results['full'][1] * W1  # placeholder to keep shapes; overlap computed on (ai, b) pairs below
            fp = torch.stack([results['full'][1], results['full'][2]], -1)[:, :10]
            lp_ = torch.stack([results['b_late'][1], results['b_late'][2]], -1)[:, :10]
            agree_b.append(((lp_[:, :, None] == fp[:, None]).all(-1).any(-1)).float().mean(-1).cpu())

    def boot(x, y, n=2000):
        d = torch.tensor(x) - torch.tensor(y)
        g = torch.Generator().manual_seed(0)
        bs = d[torch.randint(0, len(d), (n, len(d)), generator=g)].mean(1).sort().values
        return [d.mean().item(), bs[int(.025 * n)].item(), bs[int(.975 * n)].item()]

    us = sorted(per['full'])
    res = {'setup': {'gpu': torch.cuda.get_device_name(0), 'entry': e, 'train_users': len(train), 'eval_users': len(us)},
           'train_log': log, 'top10_prefix_overlap_b_late': torch.cat(agree_b).mean().item()}
    for n in per:
        res[n] = {key: statistics.mean(per[n][u][key] for u in us) for key in per[n][us[0]]}
        if n != 'full':
            for key in ('HR@1', 'HR@10', 'NDCG@10'):
                res[n][f'd{key}_vs_full'] = boot([per[n][u][key] for u in us], [per['full'][u][key] for u in us])
    print(json.dumps({'stage': 'bridge_b_eval', **res}), flush=True)
    (out / f'bridge_b_e{e}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
