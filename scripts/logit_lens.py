"""Logit lens over SID levels: at which layer is each code decided?

For each user: run the fused prompt + first-level-trie forward, and after every layer apply the
final RMSNorm and the SID rows of the LM head (codebook-renormalized) to
  (a) the last prompt position  -> distribution over valid first codes a
  (b) the a-node of the final top-1 a -> distribution over valid b under it.
Report per layer: top-1 agreement with the final layer, top-10 overlap, and HR@10 of an
"exit here" ranking of first codes (target's a in the top-10 first codes).
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights, Forward, rmsnorm
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun import data


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--users', type=int, default=500); ap.add_argument('--dtype', default='bfloat16'); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device); dt = getattr(torch, a.dtype)
    tok = data.load_tokenizer(a.release); rows = data.test_rows(a.release)
    sel = list(range(min(a.users, len(rows))))
    ids = data.prompts(a.release, tok, [rows[i] for i in sel])
    tg = [data.sid_tokens(tok, rows[i]['item_sid']) for i in sel]
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    W = Qwen2Weights(Path(a.release) / data.CKPT, dt, dev)
    fwd = Forward(W)
    head = W.embed[cat.sid_ids].float()
    loc = cat.tok2loc
    first = cat.first
    nA = len(first)
    L = W.L
    agree_a = torch.zeros(L + 1); top10_a = torch.zeros(L + 1); hit_a = torch.zeros(L + 1)
    agree_b = torch.zeros(L + 1); n = 0
    a_index = {t: i for i, t in enumerate(first.tolist())}
    for u in range(len(sel)):
        x = W.embed[torch.tensor([ids[u]], device=dev)]
        xn = W.embed[first][None]
        B, P, _ = x.shape
        T = P + nA
        lengths = torch.tensor([P], device=dev)
        pos = torch.cat([torch.arange(P, device=dev)[None], lengths[:, None].expand(1, nA)], 1)
        i = torch.arange(T, device=dev)
        causal = i[None, :] <= i[:, None]
        node_q = i[:, None] >= P
        mask = torch.where(node_q, (i[None, :] < P) | (i[None, :] == i[:, None]), causal)[None, None]
        h = torch.cat([x, xn], 1)
        rep = W.H // W.KV
        per_layer = []
        for li, layer in enumerate(W.layers):
            q, k, v = fwd.qkv(h, layer, pos)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(rep, 1),
                                               v.transpose(1, 2).repeat_interleave(rep, 1), attn_mask=mask, scale=fwd.scale)
            h = fwd.post(h, o.transpose(1, 2).reshape(1, T, W.H * W.hd), layer)
            per_layer.append(h)
        dists_a, dists_b = [], []
        for hl in per_layer:
            z = rmsnorm(hl, W.norm, W.eps).float() @ head.T                  # [1,T,600]
            la = z[0, P - 1, loc[first]]                                       # valid first codes
            dists_a.append(la)
            dists_b.append(z[0, P:])                                           # all a-nodes
        fa = dists_a[-1].argmax().item()
        ft10 = set(dists_a[-1].topk(10).indices.tolist())
        tb = cat.t1[fa][cat.v1[fa]]
        fb = dists_b[-1][fa, loc[tb]].argmax().item()
        ta = a_index.get(tg[u][0], -1)
        for li in range(L):
            agree_a[li] += dists_a[li].argmax().item() == fa
            top10_a[li] += len(set(dists_a[li].topk(10).indices.tolist()) & ft10) / 10
            hit_a[li] += ta in dists_a[li].topk(10).indices.tolist()
            agree_b[li] += dists_b[li][fa, loc[tb]].argmax().item() == fb
        n += 1
    res = {'users': n, 'layers': list(range(1, L + 1)),
           'a_top1_agree': (agree_a[:L] / n).tolist(), 'a_top10_overlap': (top10_a[:L] / n).tolist(),
           'a_target_in_top10': (hit_a[:L] / n).tolist(), 'b_top1_agree_given_a': (agree_b[:L] / n).tolist()}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / 'logit_lens.json', 'w'), indent=1)
    for li in range(0, L, 3):
        print(f"layer {li+1:2d}: a top1 agree {res['a_top1_agree'][li]:.3f}  a top10 overlap {res['a_top10_overlap'][li]:.3f}  "
              f"target a in top10 {res['a_target_in_top10'][li]:.3f}  b top1 agree {res['b_top1_agree_given_a'][li]:.3f}", flush=True)


if __name__ == '__main__':
    main()
