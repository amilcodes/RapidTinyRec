"""Exhaustive per-token log-probabilities for every catalog SID and test user.

For each user and every valid trie node this records the full-vocabulary
log-probabilities and the log-mass of the current codebook, so that any
search algorithm (beam of any width, exact/certified, prefix-filtered) under
any scoring rule (full-vocab, no-suffix, codebook-, trie-renormalized) can be
simulated offline without the model.

Output (per shard, torch.save):
  rows [U]               test-row indices
  la   [U, nA]           lp(a) full-vocab, first codes in catalog order
  cba  [U]               log-mass of level-a codebook at the prompt
  lb   [U, nA, W1]       lp(b|a) (-inf for invalid slots)
  cbb  [U, nA]           log-mass of level-b codebook after each a
  lc   [U, S]            lp(c|ab) for every unique SID (sid order = ExactEngine.sid_tok)
  cbc  [U, S]            log-mass of level-c codebook after (a,b)
  lnl  [U, S]            lp(\\n | abc);  leos [U, S] lp(EOS | abc\\n)
"""
import argparse, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactEngine
from rapidtinyrec.speedrun import data


@torch.inference_mode()
def one_user(e, ids, dev, chunk):
    W, cat, fwd = e.W, e.cat, e.fwd
    P = (len(ids) + 15) // 16 * 16
    tokens = torch.full((1, P), 151645, dtype=torch.long, device=dev)
    tokens[0, :len(ids)] = torch.tensor(ids, device=dev)
    lengths = torch.tensor([len(ids)], device=dev)
    nA = len(cat.first)
    bufs = {k: v for k, v in ExactEngine.alloc(e, 1, P).items() if k in ('kc', 'vc', 'kp2', 'vp2')}
    h = fwd.prefill(W.embed[tokens], torch.arange(P, device=dev)[None], bufs['kc'], bufs['vc'])
    lg = e._logits(h[0, len(ids) - 1][None])[0]
    lse = torch.logsumexp(lg, -1)
    la = lg[cat.first] - lse
    lvl = cat.level_mask
    sid_ids = cat.sid_ids
    cb = lambda lgx, lsex, l: torch.logsumexp(lgx[..., sid_ids[lvl[l]]], -1) - lsex
    cba = cb(lg, lse, 0)
    neg = torch.finfo(torch.float32).min
    pmask = torch.where(torch.arange(P, device=dev)[None] < lengths[:, None], 0.0, neg)
    L0 = lengths[:, None, None]
    h = fwd.step(W.embed[cat.first][None, :, None], L0.expand(1, nA, 1), bufs['kc'], bufs['vc'], pmask,
                 bufs['kp2'], bufs['vp2'], 0, lengths=lengths)
    lg = e._logits(h[0, :, 0])                       # [nA,V]
    lse = torch.logsumexp(lg, -1)
    lb = (lg.gather(-1, cat.t1) - lse[:, None]).masked_fill(~cat.v1, float('-inf'))
    cbb = cb(lg, lse, 1)
    S = e.S
    lc = torch.empty(S, device=dev); cbc = torch.empty(S, device=dev)
    lnl = torch.empty(S, device=dev); leos = torch.empty(S, device=dev)
    for s in range(0, S, chunk):
        idx = torch.arange(s, min(S, s + chunk), device=dev)
        n = len(idx)
        sid = e.sid_tok[idx][None]
        ai = e.sid_a[idx]
        kp3 = torch.zeros(W.L, 1, n, 4, W.KV, W.hd, device=dev, dtype=W.dtype)
        vp3 = torch.zeros_like(kp3)
        kp3[:, 0, :, 0] = bufs['kp2'][:, 0][:, ai, 0]
        vp3[:, 0, :, 0] = bufs['vp2'][:, 0][:, ai, 0]
        nl = torch.full_like(sid[..., 0], e.newline)
        x = W.embed[torch.stack([sid[..., 1], sid[..., 2], nl], -1)]
        pos3 = torch.cat([L0 + 1, L0 + 2, L0 + 3], -1).expand(1, n, 3)
        h = fwd.step(x, pos3, bufs['kc'], bufs['vc'], pmask, kp3, vp3, 1, lengths=lengths)
        lg = e._logits(h)[0]                         # [n,3,V]
        lse = torch.logsumexp(lg, -1)
        lc[idx] = lg[:, 0].gather(-1, sid[0, :, 2:3])[:, 0] - lse[:, 0]
        cbc[idx] = cb(lg[:, 0], lse[:, 0], 2)
        lnl[idx] = lg[:, 1, e.newline] - lse[:, 1]
        leos[idx] = lg[:, 2, e.eos] - lse[:, 2]
    return dict(la=la, cba=cba, lb=lb, cbb=cbb, lc=lc, cbc=cbc, lnl=lnl, leos=leos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(a.device)
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release)
    ids = data.prompts(a.release, tok, rows)
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, getattr(torch, a.dtype), dev)
    e = ExactEngine(W, cat, n_expand=64)
    mine = [i for i in range(len(rows)) if i % a.nshards == a.shard]
    if a.limit:
        mine = mine[:a.limit]
    acc = {k: [] for k in ('la', 'cba', 'lb', 'cbb', 'lc', 'cbc', 'lnl', 'leos')}
    t0 = time.time()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    done_rows = []
    for n, i in enumerate(mine):
        r = one_user(e, ids[i], dev, a.chunk)
        for k, v in r.items():
            acc[k].append(v.float().cpu())
        done_rows.append(i)
        if n % 100 == 0:
            print(json.dumps({'done': n, 'of': len(mine), 'elapsed_s': time.time() - t0}), flush=True)
        if (n + 1) % 400 == 0 or n + 1 == len(mine):
            save(a, e, cat, tok, rows, acc, done_rows)
    print(json.dumps({'stage': 'done', 'users': len(mine), 'elapsed_s': time.time() - t0}), flush=True)


def save(a, e, cat, tok, rows, acc, mine):
    out = {k: torch.stack(v) for k, v in acc.items()}
    out['rows'] = torch.tensor(mine)
    out['meta'] = {'sid_tok': e.sid_tok.cpu(), 'first': cat.first.cpu(), 't1': cat.t1.cpu(), 'v1': cat.v1.cpu(),
                   'sid_a': e.sid_a.cpu(), 'sid_flat': e.sid_flat.cpu(), 'sid_pair': e.sid_pair.cpu(),
                   'W1': cat.W1, 'dtype': a.dtype, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                   'targets': [data.sid_tokens(tok, rows[i]['item_sid']) for i in mine]}
    tmp = Path(a.out) / f'scores_{a.shard}_of_{a.nshards}.pt.tmp'
    torch.save(out, tmp)
    tmp.replace(Path(a.out) / f'scores_{a.shard}_of_{a.nshards}.pt')


if __name__ == '__main__':
    main()
