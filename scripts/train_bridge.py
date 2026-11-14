"""Train a late-entry bridge on validation prompts; evaluate on all test users.

usage: train_bridge.py --train-release VALID_RELEASE --release TEST_RELEASE --out DIR --entry 22

The bridge (rapidtinyrec/speedrun/bridge.py) maps a parent node's layer-e residual and the tokens
(b, a) to the prefix row's layer-e input. Stage 1: ridge fit (label-free). Stage 2: an MLP on top,
trained by distillation: KL between the full model's third-code distribution (codebook softmax,
forward 2 at full depth) and the late-entry forward's. No targets (items) are used for training.
Evaluation on the test split, certified-exact Office objective (codebook, w_c = 0.5), paired against
the full-depth second forward.
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactEngine3
from rapidtinyrec.speedrun.bridge import Bridge


def template_of(all_ids):
    lcp = list(all_ids[0])
    for ids_ in all_ids:
        k = 0
        while k < min(len(lcp), len(ids_)) and lcp[k] == ids_[k]:
            k += 1
        lcp = lcp[:k]
    return lcp


def build(release, dtype, dev, weights, tmpl_ids=None):
    tok = data.load_tokenizer(release)
    rows = data.test_rows(release)
    ids = data.prompts(release, tok, rows)
    targets = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    cat = SidCatalog(tok, Path(release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(release) / data.CKPT, dtype, dev)
    eng = ExactEngine3(W, cat, n_prefix=64, n_expand=50, suffix_mode='none', norm='codebook', fuse_first=True)
    eng.set_weights(*weights)
    eng.set_template(tmpl_ids if tmpl_ids is not None else template_of(ids))
    return tok, rows, ids, targets, cat, W, eng


class Stage:
    """Forward 1 + prefix selection + full-depth forward 2 for a chunk of users (no grad)."""

    def __init__(self, tok, ids, cat, W, eng, entry, b_bridge=None, b_entry=None):
        self.tok, self.ids, self.cat, self.W, self.eng, self.e = tok, ids, cat, W, eng, entry
        self.c_idx = torch.nonzero(cat.level_mask[2]).flatten()      # columns of sid_head for level c
        self.bb, self.eb = b_bridge, b_entry                          # cascade: late first-level rows
        assert b_bridge is None or b_entry <= entry

    @torch.no_grad()
    def __call__(self, chunk):
        W, cat, eng, fwd = self.W, self.cat, self.eng, self.eng.fwd
        dev = W.device
        B = len(chunk)
        P = (max(len(self.ids[i]) for i in chunk) + 7) // 8 * 8
        t = torch.full((B, P), self.tok.eos_token_id, dtype=torch.long)
        for j, i in enumerate(chunk):
            t[j, :len(self.ids[i])] = torch.tensor(self.ids[i])
        t = t.to(dev)
        lengths = torch.tensor([len(self.ids[i]) for i in chunk], device=dev)
        bufs = eng.alloc(B, P)
        T0 = eng._T0()
        nA, M, D = len(cat.first), eng.M, W.D
        cap = {self.e: None}
        capP = {self.eb: None} if self.bb is not None else None
        hp, hn = fwd.prefill_tree(W.embed[t[:, T0:]], W.embed[cat.first][None].expand(B, nA, D), lengths,
                                  bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=eng.tmpl, capture=cap,
                                  capture_prompt=capP)
        bi = torch.arange(B, device=dev)
        la = eng._lp(hp[bi, lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        pmask = torch.where(torch.arange(P, device=dev)[None] < lengths[:, None], 0.0, torch.finfo(torch.float32).min)
        if self.bb is not None:
            # late first-level rows: enter at eb from the last prompt token's state; their K/V (layers >= eb)
            # go to separate buffers; their state entering layer e is the parent state for the third code
            kp2l, vp2l = torch.zeros_like(bufs['kp2']), torch.zeros_like(bufs['vp2'])
            hl = capP[self.eb][bi, lengths - 1 - T0][:, None, None].expand(B, nA, 1, D)
            el = W.embed[t[bi, lengths - 1]][:, None, None].expand(B, nA, 1, D)
            ea_ = W.embed[cat.first][None, :, None].expand(B, nA, 1, D)
            x0b = self.bb(hl, ea_, el).to(W.dtype)
            capL = {self.e: None}
            if self.e == self.eb:
                capL[self.e] = x0b[:, :, 0].clone()
            hnl = fwd.step(x0b, lengths[:, None, None].expand(B, nA, 1), bufs['kc'], bufs['vc'], pmask, kp2l, vp2l, 0,
                           lengths=lengths, start=self.eb, capture=capL if self.e > self.eb else None)
            hn = hnl[:, :, 0]
            hA_src = capL[self.e] if self.e == self.eb else capL[self.e][:, :, 0]
        lb = eng._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * cat.W1)
        u2m, pf = (la.repeat_interleave(cat.W1, dim=1) + lb).topk(M, -1)
        ai, col = pf // cat.W1, pf % cat.W1
        b_tok, a_tok = cat.t1[ai, col], cat.first[ai]
        pair = cat.pair_table[ai, col]
        for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
            kp[:, :, :, 0] = kp2[:, bi[:, None], ai, 0]
        pos = (lengths[:, None, None] + 1).expand(B, M, 1)
        capB = {self.e: None}
        h = fwd.step(W.embed[b_tok][:, :, None], pos, bufs['kc'], bufs['vc'], pmask, bufs['kp3'], bufs['vp3'], 1,
                     lengths=lengths, capture=capB)
        hA = cap[self.e][bi[:, None], ai][:, :, None]
        if self.bb is not None:
            # student parents: late first-level states and K/V (teacher above used the full-depth ones)
            hA = hA_src[bi[:, None], ai][:, :, None]
            for kp, kp2 in ((bufs['kp3'], kp2l), (bufs['vp3'], vp2l)):
                kp[:, :, :, 0] = kp2[:, bi[:, None], ai, 0]
        return dict(B=B, chunk=chunk, bufs=bufs, lengths=lengths, pos=pos, pmask=pmask, u2m=u2m, ai=ai,
                    a_tok=a_tok, b_tok=b_tok, t2=cat.t2[pair], v2=cat.v2[pair], h_full=h[:, :, 0],
                    hA=hA, hB=capB[self.e])

    def late(self, st, x0):
        """Layers e..L-1 for the prefix rows starting from x0 [B,M,1,D] (differentiable)."""
        fwd = self.eng.fwd
        h = fwd.step(x0.to(self.W.dtype), st['pos'], st['bufs']['kc'], st['bufs']['vc'], st['pmask'],
                     st['bufs']['kp3'], st['bufs']['vp3'], 1, lengths=st['lengths'], start=self.e,
                     functional=True)
        return h[:, :, 0]

    def c_logp(self, h):
        """log-softmax over the c codebook [B,M,256]."""
        lg = F.linear(h, self.eng.sid_head[self.c_idx]).float()
        return lg.log_softmax(-1)

    def c_scores(self, h, st):
        return self.eng._lp(h, 2, st['t2'], st['v2']).masked_fill(~st['v2'], float('-inf'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-release', required=True); ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--entry', type=int, default=22)
    ap.add_argument('--train-users', type=int, default=3000); ap.add_argument('--ridge-users', type=int, default=600)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--hidden', type=int, default=2048)
    ap.add_argument('--eval-users', type=int, default=0, help='0 = all test users')
    ap.add_argument('--weights', default='1,0.5,0')
    ap.add_argument('--row-weight', type=float, default=0.0,
                    help='mix of prefix-probability weighting in the KL (0 = uniform over the M rows)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--b-bridge', default=None, help='first-level bridge (cascade training/eval)')
    ap.add_argument('--b-entry', type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.manual_seed(0); random.seed(0)
    wts = tuple(float(x) for x in a.weights.split(','))
    e = a.entry
    # ---------------- training on validation prompts (bf16 model)
    tok, rows, ids, _, cat, W, eng = build(a.train_release, torch.bfloat16, dev, wts)
    tmpl_ids = list(eng.template_ids)
    users = list(range(len(rows))); random.shuffle(users)
    held = users[:200]; train = users[200:200 + a.train_users]
    bb = None
    if a.b_bridge:
        sd_b = torch.load(a.b_bridge, map_location='cpu')
        bb = Bridge(W.D, sd_b['inp.weight'].shape[0]); bb.load_state_dict(sd_b); bb = bb.to(dev).eval()
    S = Stage(tok, ids, cat, W, eng, e, bb, a.b_entry)
    D = W.D
    br = Bridge(D, a.hidden).to(dev)
    sid_emb = W.embed[cat.sid_ids].float()
    br.scale.fill_(1.0 / sid_emb.pow(2).mean().sqrt().item())
    feats = 3 * D + 1
    XtX = torch.zeros(feats, feats, device=dev, dtype=torch.float64)
    XtY = torch.zeros(feats, D, device=dev, dtype=torch.float64)
    t0 = time.time()
    for k in range(0, a.ridge_users, a.batch):
        st = S(train[k:k + a.batch])
        X = torch.cat([st['hA'].float(), W.embed[st['b_tok']][:, :, None].float(), W.embed[st['a_tok']][:, :, None].float(),
                       torch.ones_like(st['hA'][..., :1]).float()], -1).reshape(-1, feats).double()
        XtX += X.T @ X
        XtY += X.T @ (st['hB'].float() - st['hA'].float()).reshape(-1, D).double()
    lam = 1e-2 * XtX.diagonal().mean()
    br.R.copy_(torch.linalg.solve(XtX + lam * torch.eye(feats, device=dev, dtype=torch.float64), XtY).float())
    del XtX, XtY
    print(json.dumps({'stage': 'ridge', 's': time.time() - t0}), flush=True)

    def held_loss():
        tot, n = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(held), a.batch):
                st = S(held[k:k + a.batch])
                tgt = S.c_logp(st['h_full'])
                x0 = br(st['hA'], W.embed[st['b_tok']][:, :, None], W.embed[st['a_tok']][:, :, None])
                lp = S.c_logp(S.late(st, x0))
                tot += F.kl_div(lp, tgt, log_target=True, reduction='sum').item(); n += lp.shape[0] * lp.shape[1]
        return tot / n

    log = [{'epoch': 0, 'held_kl': held_loss()}]
    print(json.dumps({'stage': 'train', **log[-1]}), flush=True)
    opt = torch.optim.AdamW([p for p in br.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        random.shuffle(train)
        t0 = time.time(); run = []
        for k in range(0, len(train), a.batch):
            st = S(train[k:k + a.batch])
            tgt = S.c_logp(st['h_full'])
            x0 = br(st['hA'], W.embed[st['b_tok']][:, :, None], W.embed[st['a_tok']][:, :, None])
            lp = S.c_logp(S.late(st, x0))
            kl = F.kl_div(lp, tgt, log_target=True, reduction='none').sum(-1)          # [B, M]
            M_ = kl.shape[1]
            w = (1 - a.row_weight) / M_ + a.row_weight * st['u2m'].float().softmax(-1)
            loss = (kl * w).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(br.parameters(), 1.0)
            opt.step(); sched.step()
            run.append(loss.item())
        log.append({'epoch': ep + 1, 'train_kl': statistics.mean(run), 'held_kl': held_loss(), 's': time.time() - t0})
        print(json.dumps({'stage': 'train', **log[-1]}), flush=True)
    torch.save(br.state_dict(), out / f'bridge_e{e}{a.tag}.pt')
    del W, eng, S
    torch.cuda.empty_cache()

    # ---------------- evaluation on the test split (fp16 model, same template rule as the engine)
    tok, rows, ids, targets, cat, W, eng = build(a.release, torch.float16, dev, wts)
    S = Stage(tok, ids, cat, W, eng, e)
    Sc = Stage(tok, ids, cat, W, eng, e, bb, a.b_entry) if bb is not None else S
    br.eval()
    order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    if a.eval_users:
        order = order[:a.eval_users]
    names = ('full', 'ridge', 'mlp', 'exit')
    per = {n: {} for n in names}
    agree = {n: [] for n in names}
    br_ridge = Bridge(W.D, a.hidden).to(dev); br_ridge.load_state_dict(br.state_dict())
    torch.nn.init.zeros_(br_ridge.out.weight); torch.nn.init.zeros_(br_ridge.out.bias)
    t0 = time.time()

    def ranked(st, h):
        B = st['B']
        lc = S.c_scores(h, st)
        top = (st['u2m'][..., None] + lc).view(B, -1).topk(20, -1).indices
        row = top // cat.W2
        sid = torch.stack([st['a_tok'].gather(1, row), st['b_tok'].gather(1, row), st['t2'].view(B, -1).gather(1, top)], -1)
        return [[tuple(x) for x in u] for u in sid.cpu().tolist()]

    with torch.no_grad():
        for k in range(0, len(order), a.batch):
            chunk = order[k:k + a.batch]
            st = S(chunk)
            stc = Sc(chunk) if Sc is not S else st
            eb, ea = W.embed[stc['b_tok']][:, :, None], W.embed[stc['a_tok']][:, :, None]
            lists = {'full': ranked(st, st['h_full']),
                     'ridge': ranked(stc, Sc.late(stc, br_ridge(stc['hA'], eb, ea))),
                     'mlp': ranked(stc, Sc.late(stc, br(stc['hA'], eb, ea)))}
            hx = eng.fwd.step(W.embed[st['b_tok']][:, :, None], st['pos'], st['bufs']['kc'], st['bufs']['vc'], st['pmask'],
                              st['bufs']['kp3'], st['bufs']['vp3'], 1, lengths=st['lengths'], n_layers=W.L - e)
            lists['exit'] = ranked(st, hx[:, :, 0])
            for n, L in lists.items():
                for j, i in enumerate(chunk):
                    per[n][i] = data.rank_metrics(L[j], targets[i], ks=(1, 5, 10, 20))
                    f = lists['full'][j]
                    agree[n].append(torch.tensor([float(L[j][0] == f[0]), len(set(L[j][:10]) & set(f[:10])) / 10, float('nan')]))
            agree_stack = None

    def boot(x, y, n=2000):
        d = torch.tensor(x) - torch.tensor(y)
        g = torch.Generator().manual_seed(0)
        bs = d[torch.randint(0, len(d), (n, len(d)), generator=g)].mean(1).sort().values
        return [d.mean().item(), bs[int(.025 * n)].item(), bs[int(.975 * n)].item()]

    users = sorted(per['full'])
    res = {'setup': {'gpu': torch.cuda.get_device_name(0), 'entry': e, 'b_entry': a.b_entry, 'cascade': bb is not None,
                     'train_users': len(train), 'held_users': len(held),
                     'eval_users': len(users), 'weights': a.weights, 'eval_s': time.time() - t0}, 'train_log': log}
    for n in names:
        m = {key: statistics.mean(per[n][u][key] for u in users) for key in per[n][users[0]]}
        g = torch.stack(agree[n]).nanmean(0).tolist()
        res[n] = {**m, 'top1_same_as_full': g[0], 'top10_overlap_full': g[1], 'c_argmax_agree': g[2]}
        if n != 'full':
            for key in ('HR@1', 'HR@10', 'NDCG@10'):
                res[n][f'd{key}_vs_full'] = boot([per[n][u][key] for u in users], [per['full'][u][key] for u in users])
    print(json.dumps({'stage': 'bridge_eval', **res}), flush=True)
    (out / f'bridge_e{e}{a.tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
