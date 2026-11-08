"""Certified exact top-K search over the SID trie ("beam-free" decoding).

Every scored term is a log-probability <= 0, so for an item (a, b, c)

    score(a,b,c) = lp(a) + lp(b|a) + lp(c|ab) + lp(\\n|abc) + lp(EOS|abc\\n)
                 <= U2(a,b) = lp(a) + lp(b|a).

Forward 1 (prefill) gives lp(a) for every valid first code.  Forward 2 feeds
*every* valid first code as one tree node (each node attends to the shared
prompt and itself), giving lp(b|a) for every valid (a, b) prefix, hence the
admissible bound U2 for every catalog item.  Forward 3 teacher-forces the N
items with the largest bounds as [b, c, \\n] paths and returns their exact
five-term scores.  If the K-th best exact score is >= the largest bound among
the items that were not expanded, the returned top-K equals the global top-K
of the model's full-vocabulary objective (no beam-search error).

All shapes are fixed by (B, P, N) so a request is one CUDA-graph replay.
"""
import torch

from .model import Forward
from .beam import ScoreMixin


class ExactEngine(ScoreMixin):
    def __init__(self, W, cat, n_expand=64, newline_id=198, eos_id=151645, suffix_mode='fused', ks=(1, 5, 10, 20, 50), norm='full',
                 fuse_first=False):
        self.W, self.cat, self.N = W, cat, n_expand
        self.fuse_first = fuse_first
        self.fwd = Forward(W)
        self.newline, self.eos = newline_id, eos_id
        self._init_scoring(norm)
        self.suffix_mode = suffix_mode if norm == 'full' else 'none'
        self.ks = ks
        dev = W.device
        # per-SID lookup tables (unique SIDs, sorted)
        uniq = sorted(set(cat.item_sid_tokens))
        first = cat.first.tolist()
        a_index = {a: i for i, a in enumerate(first)}
        t1 = cat.t1.tolist(); v1 = cat.v1.tolist()
        col = {}
        for i in range(len(first)):
            for j in range(cat.W1):
                if v1[i][j]:
                    col[(first[i], t1[i][j])] = j
        self.sid_tok = torch.tensor(uniq, dtype=torch.long, device=dev)            # [S,3]
        self.sid_a = torch.tensor([a_index[s[0]] for s in uniq], device=dev)       # [S]
        self.sid_flat = torch.tensor([a_index[s[0]] * cat.W1 + col[(s[0], s[1])] for s in uniq], device=dev)
        self.S = len(uniq)
        pt = cat.pair_table.tolist(); t2 = cat.t2.tolist(); v2 = cat.v2.tolist()
        sp, sc = [], []
        for s_ in uniq:
            p_ = pt[a_index[s_[0]]][col[(s_[0], s_[1])]]
            sp.append(p_)
            sc.append(next(k for k in range(cat.W2) if v2[p_][k] and t2[p_][k] == s_[2]))
        self.sid_pair = torch.tensor(sp, device=dev)
        self.sid_col = torch.tensor(sc, device=dev)
        assert self.N < self.S

    def alloc(self, B, P):
        W, dev, dt = self.W, self.W.device, self.W.dtype
        nA = len(self.cat.first)
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt)
        return dict(kc=z(W.L, B, W.KV, P, W.hd), vc=z(W.L, B, W.KV, P, W.hd),
                    kp2=z(W.L, B, nA, 1, W.KV, W.hd), vp2=z(W.L, B, nA, 1, W.KV, W.hd),
                    kp3=z(W.L, B, self.N, 4, W.KV, W.hd), vp3=z(W.L, B, self.N, 4, W.KV, W.hd))

    def _logits(self, h):
        return self.W.linear(h, self.W.lm_head)[..., :self.W.V].float()

    def run(self, tokens, lengths, bufs):
        W, cat, fwd, N = self.W, self.cat, self.fwd, self.N
        B, P = tokens.shape
        dev = tokens.device
        nA = len(cat.first)
        bidx = torch.arange(B, device=dev)[:, None]
        neg = torch.finfo(torch.float32).min
        pmask = torch.where(torch.arange(P, device=dev)[None, :] < lengths[:, None], 0.0, neg)
        L0 = lengths[:, None, None]
        if self.fuse_first:
            # ---- forward 1: prompt + every first code as tree nodes -> lp(a), lp(b|a)
            T0 = self._T0()
            hp, hn = fwd.prefill_tree(W.embed[tokens[:, T0:]], W.embed[cat.first][None].expand(B, nA, W.D), lengths,
                                      bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=self.tmpl)
            la = self._lp(hp[torch.arange(B, device=dev), lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        else:
            # ---- forward 1: prefill -> lp(a) for every valid first code
            pos = torch.arange(P, device=dev)[None, :].expand(B, P)
            h = fwd.prefill(W.embed[tokens], pos, bufs['kc'], bufs['vc'])
            la = self._lp(h[torch.arange(B, device=dev), lengths - 1], 0, cat.first[None].expand(B, -1), None)  # [B,nA]
            # ---- forward 2: every first code as a tree node -> lp(b|a)
            x = W.embed[cat.first][None, :, None].expand(B, nA, 1, W.D)
            h = fwd.step(x, L0.expand(B, nA, 1), bufs['kc'], bufs['vc'], pmask, bufs['kp2'], bufs['vp2'], 0, lengths=lengths)
            hn = h[:, :, 0]
        lb = self._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * cat.W1)
        u2 = la.repeat_interleave(cat.W1, dim=1) + lb                          # [B,nA*W1]
        bound = u2[:, self.sid_flat]                                          # [B,S]
        top, sel = bound.topk(N + 1, -1)                                      # [B,N+1]
        next_bound = top[:, N]
        sel = sel[:, :N]
        sid = self.sid_tok[sel]                                               # [B,N,3]
        ai = self.sid_a[sel]                                                  # [B,N]
        base = top[:, :N]                                                     # lp(a)+lp(b|a)
        # ---- forward 3: teacher-force [b, c, \n] for the N best-bounded items
        for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
            kp[:, :, :, 0] = kp2[:, bidx, ai, 0]
        if self.suffix_mode == 'none':
            x = W.embed[sid[..., 1]][:, :, None]
            h = fwd.step(x, (L0 + 1).expand(B, N, 1), bufs['kc'], bufs['vc'], pmask, bufs['kp3'], bufs['vp3'], 1, lengths=lengths)
            if self.norm == 'trie':
                # need all valid siblings of c for the renormalizer
                pr = self.sid_pair[sel]
                lcs = self._lp(h[:, :, 0], 2, cat.t2[pr], cat.v2[pr])
                lc = lcs.gather(-1, self.sid_col[sel][..., None])[..., 0]
            else:
                lc = self._lp(h[:, :, 0], 2, sid[..., 2:3], None)[..., 0]
            exact = base + lc
            tr = torch.stack([lc], -1)
        else:
            nl = torch.full_like(sid[..., 0], self.newline)
            x = W.embed[torch.stack([sid[..., 1], sid[..., 2], nl], -1)]      # [B,N,3,D]
            pos3 = torch.cat([L0 + 1, L0 + 2, L0 + 3], -1).expand(B, N, 3)
            h = fwd.step(x, pos3, bufs['kc'], bufs['vc'], pmask, bufs['kp3'], bufs['vp3'], 1, lengths=lengths)
            lg = self._logits(h)                                              # [B,N,3,V]
            lse = torch.logsumexp(lg, -1)
            lc = lg[..., 0, :].gather(-1, sid[..., 2:3])[..., 0] - lse[..., 0]
            ln = lg[..., 1, self.newline] - lse[..., 1]
            le = lg[..., 2, self.eos] - lse[..., 2]
            exact = base + lc + self.suffix_w * (ln + le)
            tr = torch.stack([lc, ln, le], -1)
        final, order = torch.sort(exact, dim=-1, descending=True, stable=True)
        cert = torch.stack([final[:, k - 1] >= next_bound for k in self.ks if k <= N], -1)
        return {'sids': sid.gather(1, order[..., None].expand(B, N, 3)), 'scores': final,
                'certified': cert, 'next_bound': next_bound,
                'trans': tr.gather(1, order[..., None].expand(B, N, tr.shape[-1]))}

    @torch.inference_mode()
    def exhaustive(self, tokens, lengths, chunk=512):
        """Offline reference: exact scores of every unique SID (eager, chunked), any normalization."""
        W, cat, fwd = self.W, self.cat, self.fwd
        B, P = tokens.shape
        assert B == 1
        dev = tokens.device
        nA = len(cat.first)
        bufs = {k: v for k, v in ExactEngine.alloc(self, 1, P).items() if k in ('kc', 'vc', 'kp2', 'vp2')}
        pos = torch.arange(P, device=dev)[None]
        h = fwd.prefill(W.embed[tokens], pos, bufs['kc'], bufs['vc'])
        la = self._lp(h[0, lengths[0] - 1][None], 0, cat.first[None], None)
        neg = torch.finfo(torch.float32).min
        pmask = torch.where(torch.arange(P, device=dev)[None] < lengths[:, None], 0.0, neg)
        L0 = lengths[:, None, None]
        x = W.embed[cat.first][None, :, None]
        h = fwd.step(x, L0.expand(1, nA, 1), bufs['kc'], bufs['vc'], pmask, bufs['kp2'], bufs['vp2'], 0, lengths=lengths)
        lb = self._lp(h[:, :, 0], 1, cat.t1[None], cat.v1[None])
        u2 = (la.repeat_interleave(cat.W1, 1) + lb.reshape(1, -1))[0, self.sid_flat]
        scores = torch.empty(self.S, device=dev)
        full = self.norm == 'full' and self.suffix_mode != 'none'
        for s in range(0, self.S, chunk):
            idx = torch.arange(s, min(self.S, s + chunk), device=dev)
            n = len(idx)
            sid = self.sid_tok[idx][None]
            ai = self.sid_a[idx][None]
            kp3 = torch.zeros(W.L, 1, n, 4, W.KV, W.hd, device=dev, dtype=W.dtype)
            vp3 = torch.zeros_like(kp3)
            kp3[:, :, :, 0] = bufs['kp2'][:, 0][:, ai[0], 0][:, None]
            vp3[:, :, :, 0] = bufs['vp2'][:, 0][:, ai[0], 0][:, None]
            if full:
                nl = torch.full_like(sid[..., 0], self.newline)
                x = W.embed[torch.stack([sid[..., 1], sid[..., 2], nl], -1)]
                pos3 = torch.cat([L0 + 1, L0 + 2, L0 + 3], -1).expand(1, n, 3)
                h = fwd.step(x, pos3, bufs['kc'], bufs['vc'], pmask, kp3, vp3, 1, lengths=lengths)
                lg = self._logits(h)
                lse = torch.logsumexp(lg, -1)
                lc = (lg[..., 0, :].gather(-1, sid[..., 2:3])[..., 0] - lse[..., 0]) * self.level_w[2]
                ex = u2[idx] + lc[0] + self.suffix_w * (lg[0, :, 1, self.newline] - lse[0, :, 1] + lg[0, :, 2, self.eos] - lse[0, :, 2])
            else:
                x = W.embed[sid[..., 1]][:, :, None]
                h = fwd.step(x, (L0 + 1).expand(1, n, 1), bufs['kc'], bufs['vc'], pmask, kp3, vp3, 1, lengths=lengths)
                pr = self.sid_pair[idx][None]
                if self.norm == 'trie':
                    lcs = self._lp(h[:, :, 0], 2, cat.t2[pr], cat.v2[pr])
                    lc = lcs.gather(-1, self.sid_col[idx][None, :, None])[..., 0]
                else:
                    lc = self._lp(h[:, :, 0], 2, sid[..., 2:3], None)[..., 0]
                ex = u2[idx] + lc[0]
            scores[idx] = ex
        return scores, u2


class ExactEngine3(ExactEngine):
    """Three-stage certified search with a tight c-level bound.

    F1 prefill -> lp(a).  F2 all first codes -> lp(b|a) -> U2 per prefix.
    F3 the M best prefixes, one token each (b) -> lp(c|ab) for all children
       -> U3 = U2 + lp(c|ab), exact if the scoring has no forced suffix.
    F4 (full-vocab scoring only) the N best items by U3, tokens [c, \\n] ->
       lp(\\n), lp(EOS) -> exact five-term scores.
    Certificate: K-th exact score >= max(U3 of the best un-suffixed item,
    U2 of the best unexpanded prefix).
    """

    def __init__(self, W, cat, n_prefix=128, n_expand=32, fuse_first=False, exit_c=None, **kw):
        super().__init__(W, cat, n_expand=n_expand, **kw)
        self.M = n_prefix
        self.fuse_first = fuse_first
        self.exit_c = exit_c   # early exit (layer count) for the c-level forward; no-suffix scoring only
        assert exit_c is None or self.suffix_mode == 'none'
        self.N = n_expand if self.suffix_mode != 'none' else min(n_expand, n_prefix * cat.W2)
        self.late, self.bridge = None, None

    def set_late_entry(self, entry, bridge):
        """Third-code forward runs only layers entry..L-1; its input is bridge(parent residual at
        `entry`, E[b], E[a]) (rapidtinyrec/speedrun/bridge.py). Needs fuse_first and no forced suffix."""
        assert self.fuse_first and self.suffix_mode == 'none' and self.exit_c is None
        self.late, self.bridge = int(entry), bridge.to(self.W.device).eval().requires_grad_(False)
        if self.W.dtype in (torch.float16, torch.bfloat16):
            self.bridge.half_(self.W.dtype)

    node_late_cfg = None

    def set_node_late(self, entry, bridge):
        """First-level trie nodes join forward 1 at `entry` (bridge from the last prompt token's state)."""
        assert self.fuse_first
        assert self.late is None or self.late >= entry, 'third-code entry must be at or after the node entry'
        br = bridge.to(self.W.device).eval().requires_grad_(False)
        if self.W.dtype in (torch.float16, torch.bfloat16):
            br.half_(self.W.dtype)
        self.node_late_cfg = (int(entry), br)

    def _node_late(self, tokens, lengths):
        if self.node_late_cfg is None:
            return None
        e, br = self.node_late_cfg
        B = tokens.shape[0]
        last = self.W.embed[tokens[torch.arange(B, device=tokens.device), lengths - 1]]
        return dict(entry=e, bridge=br, last_emb=last)

    def _f2_input(self, cap, ai, b_tok):
        """Input rows for forward 2: token embeddings, or bridged parent states (late entry)."""
        W = self.W
        if self.late is None:
            return W.embed[b_tok][:, :, None], 0
        B = ai.shape[0]
        ha = cap[self.late][torch.arange(B, device=ai.device)[:, None], ai][:, :, None]
        x0 = self.bridge(ha, W.embed[b_tok][:, :, None], W.embed[self.cat.first[ai]][:, :, None])
        return x0.to(W.dtype), self.late

    def alloc(self, B, P):
        W, dev, dt = self.W, self.W.device, self.W.dtype
        nA = len(self.cat.first)
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt)
        return dict(kc=z(W.L, B, W.KV, P, W.hd), vc=z(W.L, B, W.KV, P, W.hd),
                    kp2=z(W.L, B, nA, 1, W.KV, W.hd), vp2=z(W.L, B, nA, 1, W.KV, W.hd),
                    kp3=z(W.L, B, self.M, 2, W.KV, W.hd), vp3=z(W.L, B, self.M, 2, W.KV, W.hd),
                    kp4=z(W.L, B, self.N, 4, W.KV, W.hd), vp4=z(W.L, B, self.N, 4, W.KV, W.hd))

    def run(self, tokens, lengths, bufs):
        W, cat, fwd, M, N = self.W, self.cat, self.fwd, self.M, self.N
        B, P = tokens.shape
        dev = tokens.device
        nA = len(cat.first)
        bidx = torch.arange(B, device=dev)[:, None]
        neg = torch.finfo(torch.float32).min
        pmask = torch.where(torch.arange(P, device=dev)[None, :] < lengths[:, None], 0.0, neg)
        L0 = lengths[:, None, None]
        cap = None if self.late is None else {self.late: None}
        if self.fuse_first:
            T0 = self._T0()
            hp, hn = fwd.prefill_tree(W.embed[tokens[:, T0:]], W.embed[cat.first][None].expand(B, nA, W.D), lengths,
                                      bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=self.tmpl, capture=cap,
                                      node_late=self._node_late(tokens, lengths))
            la = self._lp(hp[torch.arange(B, device=dev), lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        else:
            pos = torch.arange(P, device=dev)[None, :].expand(B, P)
            h = fwd.prefill(W.embed[tokens], pos, bufs['kc'], bufs['vc'])
            la = self._lp(h[torch.arange(B, device=dev), lengths - 1], 0, cat.first[None].expand(B, -1), None)
            # F2
            x = W.embed[cat.first][None, :, None].expand(B, nA, 1, W.D)
            h = fwd.step(x, L0.expand(B, nA, 1), bufs['kc'], bufs['vc'], pmask, bufs['kp2'], bufs['vp2'], 0, lengths=lengths)
            hn = h[:, :, 0]
        lb = self._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], float('-inf')).view(B, nA * cat.W1)
        u2 = la.repeat_interleave(cat.W1, dim=1) + lb
        top2, pf = u2.topk(M + 1, -1)                     # prefixes (flat a*W1+j)
        bound_pref = top2[:, M]
        pf, u2m = pf[:, :M], top2[:, :M]
        ai, col = pf // cat.W1, pf % cat.W1
        b_tok = cat.t1[ai, col]
        pair = cat.pair_table[ai, col]
        # F3: one token (b) per prefix; private slot 0 = a's K/V from F2
        for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
            kp[:, :, :, 0] = kp2[:, bidx, ai, 0]
        x2, start = self._f2_input(cap, ai, b_tok)
        h = fwd.step(x2, (L0 + 1).expand(B, M, 1), bufs['kc'], bufs['vc'], pmask,
                     bufs['kp3'], bufs['vp3'], 1, lengths=lengths, n_layers=self.exit_c, start=start)
        t2, v2 = cat.t2[pair], cat.v2[pair]                # [B,M,W2]
        lc = self._lp(h[:, :, 0], 2, t2, v2).masked_fill(~v2, float('-inf'))
        u3 = (u2m[..., None] + lc).view(B, M * cat.W2)
        top3, it = u3.topk(N + 1 if self.suffix_mode != 'none' else N, -1)
        row, k = it // cat.W2, it % cat.W2
        c_tok = t2.view(B, -1).gather(1, it)
        sid = torch.stack([cat.first[ai.gather(1, row)], b_tok.gather(1, row), c_tok], -1)[:, :N]
        if self.suffix_mode == 'none':
            exact = top3[:, :N]
            next_bound = bound_pref
            if N < M * cat.W2:
                pass  # items beyond N among expanded prefixes are bounded by top3[N-1] <= exact; cert uses bound_pref
        else:
            next_bound = torch.maximum(top3[:, N], bound_pref)
            row, top3 = row[:, :N], top3[:, :N]
            for kp, kp3 in ((bufs['kp4'], bufs['kp3']), (bufs['vp4'], bufs['vp3'])):
                kp[:, :, :, 0:2] = kp3[:, bidx, row, 0:2]
            nl = torch.full_like(c_tok[:, :N], self.newline)
            x = W.embed[torch.stack([c_tok[:, :N], nl], -1)]
            pos4 = torch.cat([L0 + 2, L0 + 3], -1).expand(B, N, 2)
            h = fwd.step(x, pos4, bufs['kc'], bufs['vc'], pmask, bufs['kp4'], bufs['vp4'], 2, lengths=lengths)
            lg = self._logits(h)
            lse = torch.logsumexp(lg, -1)
            ln = lg[..., 0, self.newline] - lse[..., 0]
            le = lg[..., 1, self.eos] - lse[..., 1]
            exact = top3 + self.suffix_w * (ln + le)
        final, order = torch.sort(exact, dim=-1, descending=True, stable=True)
        cert = torch.stack([final[:, kk - 1] >= next_bound for kk in self.ks if kk <= N], -1)
        return {'sids': sid.gather(1, order[..., None].expand(B, N, 3)), 'scores': final,
                'certified': cert, 'next_bound': next_bound}


class ExactTrieEngine(ExactEngine3):
    """Certified top-K under trie-normalized scoring that never expands single-child prefixes.

    Under trie normalization the third-code term is a log-softmax over the valid children of
    (a, b), so it is exactly 0 when (a, b) has one child: every item under a single-child prefix
    is exactly scored by forward 1 (prompt + all first codes). Forward 2 expands only the M best
    multi-child prefixes. Candidate pool = children of expanded prefixes (exact) + all
    single-child items (exact); certificate: K-th pool score >= best unexpanded multi-child U2.
    Cost: two forwards for any K; forward 2 carries M rows (about 0.4 K for top-K).
    """

    def __init__(self, W, cat, n_prefix=16, n_out=50, fuse_first=True, ks=(1, 5, 10, 20, 50), norm='trie', **kw):
        kw.pop('suffix_mode', None)
        assert norm in ('trie', 'cbtrie'), 'single-child skipping needs a trie-normalized third code'
        super().__init__(W, cat, n_prefix=n_prefix, n_expand=n_out, fuse_first=fuse_first, norm=norm,
                         suffix_mode='none', ks=ks, **kw)
        self.N = n_out
        nA = len(cat.first)
        nchild = cat.v2.sum(-1)                                   # [pairs]
        valid = cat.v1.view(-1)
        nflat = torch.where(valid, nchild[cat.pair_table.view(-1)], torch.zeros_like(valid, dtype=nchild.dtype))
        self.multi_flat = nflat >= 2                              # [nA*W1]
        self.single_flat = nflat == 1
        self.single_c = cat.t2[cat.pair_table.view(-1), 0]        # the only child (valid where single_flat)
        self.flat_a = cat.first.repeat_interleave(cat.W1)         # [nA*W1] first-code token
        self.flat_b = cat.t1.view(-1)
        assert int(self.single_flat.sum()) + self.M * cat.W2 >= self.N

    def alloc(self, B, P):
        W, dev, dt = self.W, self.W.device, self.W.dtype
        nA = len(self.cat.first)
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt)
        return dict(kc=z(W.L, B, W.KV, P, W.hd), vc=z(W.L, B, W.KV, P, W.hd),
                    kp2=z(W.L, B, nA, 1, W.KV, W.hd), vp2=z(W.L, B, nA, 1, W.KV, W.hd),
                    kp3=z(W.L, B, self.M, 2, W.KV, W.hd), vp3=z(W.L, B, self.M, 2, W.KV, W.hd))

    def run(self, tokens, lengths, bufs):
        W, cat, fwd, M, N = self.W, self.cat, self.fwd, self.M, self.N
        B, P = tokens.shape
        dev = tokens.device
        nA = len(cat.first)
        bidx = torch.arange(B, device=dev)[:, None]
        neg_inf = float('-inf')
        pmask = torch.where(torch.arange(P, device=dev)[None, :] < lengths[:, None], 0.0, torch.finfo(torch.float32).min)
        L0 = lengths[:, None, None]
        T0 = self._T0()
        cap = None if self.late is None else {self.late: None}
        hp, hn = fwd.prefill_tree(W.embed[tokens[:, T0:]], W.embed[cat.first][None].expand(B, nA, W.D), lengths,
                                  bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], tmpl=self.tmpl, capture=cap,
                                      node_late=self._node_late(tokens, lengths))
        la = self._lp(hp[torch.arange(B, device=dev), lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        lb = self._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], neg_inf).view(B, nA * cat.W1)
        u2 = la.repeat_interleave(cat.W1, dim=1) + lb             # [B, nA*W1]
        top2, pf = u2.masked_fill(~self.multi_flat[None], neg_inf).topk(M + 1, -1)
        bound_pref = top2[:, M]
        pf, u2m = pf[:, :M], top2[:, :M]
        ai, col = pf // cat.W1, pf % cat.W1
        b_tok = cat.t1[ai, col]
        pair = cat.pair_table[ai, col]
        for kp, kp2 in ((bufs['kp3'], bufs['kp2']), (bufs['vp3'], bufs['vp2'])):
            kp[:, :, :, 0] = kp2[:, bidx, ai, 0]
        x2, start = self._f2_input(cap, ai, b_tok)
        h = fwd.step(x2, (L0 + 1).expand(B, M, 1), bufs['kc'], bufs['vc'], pmask,
                     bufs['kp3'], bufs['vp3'], 1, lengths=lengths, n_layers=self.exit_c, start=start)
        t2, v2 = cat.t2[pair], cat.v2[pair]
        lc = self._lp(h[:, :, 0], 2, t2, v2).masked_fill(~v2, neg_inf)
        u3 = (u2m[..., None] + lc).view(B, M * cat.W2)
        single = u2.masked_fill(~self.single_flat[None], neg_inf)
        pool = torch.cat([u3, single], 1)
        final, idx = pool.topk(N, -1)
        is_exp = idx < M * cat.W2
        e_idx = idx.clamp(max=M * cat.W2 - 1)
        row = e_idx // cat.W2
        s_idx = (idx - M * cat.W2).clamp(min=0)
        a_tok = torch.where(is_exp, cat.first[ai.gather(1, row)], self.flat_a[s_idx])
        b_out = torch.where(is_exp, b_tok.gather(1, row), self.flat_b[s_idx])
        c_out = torch.where(is_exp, t2.reshape(B, -1).gather(1, e_idx), self.single_c[s_idx])
        sid = torch.stack([a_tok, b_out, c_out], -1)
        cert = torch.stack([final[:, kk - 1] >= bound_pref for kk in self.ks if kk <= N], -1)
        return {'sids': sid, 'scores': final, 'certified': cert, 'next_bound': bound_pref}


class OnePassTrieEngine(ExactTrieEngine):
    """Every catalog item scored in ONE forward (trie scoring, late entry).

    Forward 1 carries the prompt and every first code a (as before); at layer `entry` every
    multi-child prefix (a,b) of the catalog joins as an extra row whose input is the bridge applied
    to its parent's residual at that layer; it attends to the prompt, its parent and itself.
    Single-child items need no third-code term (trie scoring), so after this one forward every
    item has its exact (late-entry) score: the top-N is the exhaustive optimum, no certificate needed.
    Rows in the last L-entry layers: prompt tail + |A| + #multi-child prefixes (Office 88 + 581).
    """

    def __init__(self, W, cat, n_out=50, **kw):
        kw.pop('n_prefix', None)
        super().__init__(W, cat, n_prefix=1, n_out=n_out, **kw)
        flat = torch.nonzero(self.multi_flat).flatten()           # flat (a*W1 + j) of multi-child prefixes
        self.mp_flat = flat
        self.mp_a = flat // cat.W1
        self.mp_b = cat.t1.view(-1)[flat]
        self.mp_pair = cat.pair_table.view(-1)[flat]
        self.R = flat.numel()

    def alloc(self, B, P):
        bufs = super().alloc(B, P)
        W, dev, dt = self.W, self.W.device, self.W.dtype
        z = lambda *s: torch.zeros(*s, device=dev, dtype=dt)
        bufs['kpr'] = z(W.L, B, self.R, 2, W.KV, W.hd)
        bufs['vpr'] = z(W.L, B, self.R, 2, W.KV, W.hd)
        return bufs

    def run(self, tokens, lengths, bufs):
        W, cat, fwd, N = self.W, self.cat, self.fwd, self.N
        assert self.late is not None and fwd.tree_fast and fwd.attn == 'triton'
        B, P = tokens.shape
        dev = tokens.device
        nA = len(cat.first)
        neg_inf = float('-inf')
        T0 = self._T0()
        late = dict(entry=self.late, bridge=self.bridge, parent=self.mp_a, b_tok=self.mp_b,
                    a_tok=cat.first[self.mp_a], kp=bufs['kpr'], vp=bufs['vpr'])
        hp, hn, hr = fwd._prefill_tree_fast(W.embed[tokens[:, T0:]], W.embed[cat.first][None].expand(B, nA, W.D), lengths,
                                            bufs['kc'], bufs['vc'], bufs['kp2'], bufs['vp2'], self.tmpl, None, None, late=late)
        la = self._lp(hp[torch.arange(B, device=dev), lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
        lb = self._lp(hn, 1, cat.t1[None].expand(B, nA, cat.W1), cat.v1[None].expand(B, nA, cat.W1))
        lb = lb.masked_fill(~cat.v1[None], neg_inf).view(B, nA * cat.W1)
        u2 = la.repeat_interleave(cat.W1, dim=1) + lb             # [B, nA*W1]
        t2, v2 = cat.t2[self.mp_pair][None].expand(B, -1, -1), cat.v2[self.mp_pair][None].expand(B, -1, -1)
        lc = self._lp(hr, 2, t2, v2).masked_fill(~v2, neg_inf)    # [B, R, W2]
        u3 = (u2[:, self.mp_flat][..., None] + lc).reshape(B, -1)
        single = u2.masked_fill(~self.single_flat[None], neg_inf)
        pool = torch.cat([u3, single], 1)
        final, idx = pool.topk(N, -1)
        RW = self.R * cat.W2
        is_r = idx < RW
        r_idx = idx.clamp(max=RW - 1)
        row = r_idx // cat.W2
        s_idx = (idx - RW).clamp(min=0)
        a_tok = torch.where(is_r, cat.first[self.mp_a[row]], self.flat_a[s_idx])
        b_out = torch.where(is_r, self.mp_b[row], self.flat_b[s_idx])
        c_out = torch.where(is_r, t2.reshape(B, -1).gather(1, r_idx), self.single_c[s_idx])
        sid = torch.stack([a_tok, b_out, c_out], -1)
        cert = torch.ones(B, len([k for k in self.ks if k <= N]), dtype=torch.bool, device=dev)
        return {'sids': sid, 'scores': final, 'certified': cert}
