"""Collaborative value term computed inside the decoder's CUDA graph.

`SASRecHead` reimplements the inference path of `scripts/sasrec.py::SASRec` (pre-norm Transformer encoder,
timeline mask, causal attention without a key-padding mask, last position, dot product with the item table)
with plain tensor ops so it can be captured, and turns item logits into the per-SID value term used by
certified search:  bonus[s] = beta * max(log q(s | history), floor),  q(s) = sum of q(i) over items i with SID s.

The engine runs it on a side stream that forks from the request stream and joins before the first use of the
bonus (after forward 1), so its latency hides behind the LLM's first forward.
"""
import ast, csv, math
from pathlib import Path

import torch
import torch.nn.functional as F


def item_to_sid(release, tok, sid_tok, n_items):
    """LongTensor [n_items]: column of each item id's SID in `sid_tok` order; items without a SID -> S (dropped)."""
    from . import data
    lines = [l.split('\t') for l in open(Path(release) / data.CATALOG).read().splitlines() if l.strip()]
    col = {tuple(s): j for j, s in enumerate(sid_tok.tolist())}
    m = torch.full((n_items,), len(col), dtype=torch.long)
    for sid, _, iid in lines:
        m[int(iid)] = col[tuple(data.sid_tokens(tok, sid))]
    return m


def history_ids(rows, L):
    """[len(rows), L] left-padded SASRec inputs (item id + 1; 0 = pad), as in scripts/sasrec.py."""
    out = torch.zeros(len(rows), L, dtype=torch.long)
    for r, row in enumerate(rows):
        h = [int(x) + 1 for x in ast.literal_eval(row['history_item_id'])][-L:]
        if h:
            out[r, L - len(h):] = torch.tensor(h)
    return out


class SASRecHead:
    def __init__(self, state, item2sid, n_sids, beta, floor=-30.0, heads=2, device='cuda', dtype=torch.float32):
        g = lambda k: state[k].to(device, dtype)
        self.item = g('item.weight')                                    # [n+1, d], row 0 = pad
        self.pos = g('pos.weight')                                      # [L, d]
        self.L, self.d = self.pos.shape
        self.heads = heads
        self.layers = []
        i = 0
        while f'enc.layers.{i}.self_attn.in_proj_weight' in state:
            p = f'enc.layers.{i}.'
            self.layers.append({k: g(p + k) for k in (
                'self_attn.in_proj_weight', 'self_attn.in_proj_bias', 'self_attn.out_proj.weight', 'self_attn.out_proj.bias',
                'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias',
                'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias')})
            i += 1
        self.norm_w, self.norm_b = g('norm.weight'), g('norm.bias')
        self.eps = 1e-5
        self.item2sid = item2sid.to(device)
        self.S = n_sids
        self.beta, self.floor = float(beta), float(floor)
        self.causal = torch.triu(torch.full((self.L, self.L), float('-inf'), device=device, dtype=dtype), 1)
        self.scale = 1.0 / math.sqrt(self.d // heads)

    @classmethod
    def from_dir(cls, path, release, tok, sid_tok, beta, device='cuda', **kw):
        state = torch.load(Path(path) / 'sasrec.pt', map_location='cpu')
        n = state['item.weight'].shape[0] - 1
        return cls(state, item_to_sid(release, tok, sid_tok.cpu(), n), len(sid_tok), beta, device=device, **kw)

    def logits(self, seq):
        """seq [B, L] (0 = pad) -> item logits [B, n] (column j = item id j). Only the last position is read,
        so the last layer computes its query, attention output and FFN for that position alone."""
        B, L = seq.shape
        H, dh = self.heads, self.d // self.heads
        x = self.item[seq] * math.sqrt(self.d) + self.pos[None, :L]
        x = x * (seq != 0)[..., None].to(x.dtype)
        nl = len(self.layers)
        for li, ly in enumerate(self.layers):
            last = li == nl - 1
            h = F.layer_norm(x, (self.d,), ly['norm1.weight'], ly['norm1.bias'], self.eps)
            q, k, v = F.linear(h, ly['self_attn.in_proj_weight'], ly['self_attn.in_proj_bias']).view(B, L, 3, H, dh).unbind(2)
            if last:
                q, x = q[:, -1:], x[:, -1:]
                att = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
            else:
                att = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale + self.causal[:L, :L]
            o = torch.einsum('bhqk,bkhd->bqhd', att.softmax(-1), v).reshape(B, x.shape[1], self.d)
            x = x + F.linear(o, ly['self_attn.out_proj.weight'], ly['self_attn.out_proj.bias'])
            h = F.layer_norm(x, (self.d,), ly['norm2.weight'], ly['norm2.bias'], self.eps)
            x = x + F.linear(F.gelu(F.linear(h, ly['linear1.weight'], ly['linear1.bias'])), ly['linear2.weight'], ly['linear2.bias'])
        h = F.layer_norm(x[:, -1], (self.d,), self.norm_w, self.norm_b, self.eps)
        return h @ self.item[1:].T

    def logq(self, seq):
        """[B, S] log q(SID | history): logsumexp of the item log-softmax over items sharing a SID."""
        p = self.logits(seq).float().softmax(-1)
        q = torch.zeros(seq.shape[0], self.S + 1, device=p.device).index_add_(1, self.item2sid, p)
        return q[:, :self.S].log()

    def bonus(self, seq):
        return self.logq(seq).clamp(min=self.floor) * self.beta

    _fn = None

    def compile(self):
        """torch.compile the whole head. Experimental: inside the engine's CUDA graph (side stream) it returned
        wrong values on H100 (runs/collabO2), so bench.py does not use it by default."""
        self._fn = torch.compile(self.bonus, dynamic=False, fullgraph=True)
        return self

    def __call__(self, seq, out=None):
        b = (self._fn or self.bonus)(seq)
        return b if out is None else out.copy_(b)
