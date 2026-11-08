"""Device-resident constrained beam search over 3-token semantic IDs.

Semantics follow MiniOneRec's released decoding: every scored token uses a
full-vocabulary log-softmax; invalid continuations are masked after
normalization; beam scores are fp32 sums; the forced newline and EOS tokens are
scored and can reorder the final 50. The two forced tokens are scored in one
two-token forward instead of two one-token forwards.
"""
import collections
import re

import torch

from .model import Forward


class SidCatalog:
    def __init__(self, tok, catalog_file, device):
        lines = [l for l in open(catalog_file).read().splitlines() if l.strip()]
        sids = [l.split('\t')[0].strip() for l in lines]
        toks = []
        for s in sids:
            parts = re.findall(r'<[a-z]_\d+>', s)
            ids = tok.convert_tokens_to_ids(parts)
            assert len(ids) == 3 and all(isinstance(i, int) and i >= 0 for i in ids), s
            toks.append(tuple(ids))
        self.item_sid_tokens = toks
        uniq = sorted(set(toks))
        self.n_items = len(sids)
        self.n_sids = len(uniq)
        first = sorted({t[0] for t in uniq})
        a_index = {a: i for i, a in enumerate(first)}
        ch1 = collections.defaultdict(set)
        ch2 = collections.defaultdict(set)
        for a, b, c in uniq:
            ch1[a].add(b)
            ch2[(a, b)].add(c)
        W1 = max(len(v) for v in ch1.values())
        W2 = max(len(v) for v in ch2.values())
        t1 = torch.zeros(len(first), W1, dtype=torch.long)
        v1 = torch.zeros(len(first), W1, dtype=torch.bool)
        pairs = []
        pair_id = {}
        pt = torch.zeros(len(first), W1, dtype=torch.long)
        for i, a in enumerate(first):
            bs = sorted(ch1[a])
            for j, b in enumerate(bs):
                t1[i, j] = b; v1[i, j] = True
                pair_id[(a, b)] = len(pairs); pairs.append((a, b))
                pt[i, j] = pair_id[(a, b)]
            t1[i, len(bs):] = bs[0]
        t2 = torch.zeros(len(pairs), W2, dtype=torch.long)
        v2 = torch.zeros(len(pairs), W2, dtype=torch.bool)
        for p, (a, b) in enumerate(pairs):
            cs = sorted(ch2[(a, b)])
            for k, c in enumerate(cs):
                t2[p, k] = c; v2[p, k] = True
            t2[p, len(cs):] = cs[0]
        self.first = torch.tensor(first, dtype=torch.long, device=device)
        self.t1, self.v1, self.pair_table = t1.to(device), v1.to(device), pt.to(device)
        self.t2, self.v2 = t2.to(device), v2.to(device)
        self.W1, self.W2 = W1, W2
        # every SID token in the vocabulary, grouped by level (codebook), for restricted heads
        vocab = tok.get_vocab()
        levels = sorted({re.match(r'<([a-z])_', t).group(1) for t in vocab if re.fullmatch(r'<[a-z]_\d+>', t)})
        sid_ids, lvl = [], []
        for li, L in enumerate(levels):
            ids_l = sorted(i for t, i in vocab.items() if re.fullmatch(rf'<{L}_\d+>', t))
            sid_ids += ids_l; lvl += [li] * len(ids_l)
        self.sid_ids = torch.tensor(sid_ids, dtype=torch.long, device=device)
        self.level_mask = torch.stack([torch.tensor(lvl, device=device) == li for li in range(len(levels))])
        tok2loc = torch.zeros(max(vocab.values()) + 1, dtype=torch.long)
        tok2loc[torch.tensor(sid_ids)] = torch.arange(len(sid_ids))
        self.tok2loc = tok2loc.to(device)
        self.stats = {'sid_vocab': len(sid_ids), 'levels': levels, 'items': self.n_items, 'unique_sids': self.n_sids, 'level1': len(first),
                      'level2': len(pairs), 'max_children_1': W1, 'max_children_2': W2}


class ScoreMixin:
    """Per-step log-probabilities under one of three normalizations.

    full      log-softmax over the whole vocabulary (MiniOneRec / HF semantics)
    codebook  log-softmax over the current level's SID tokens only (FlashRec-style)
    trie      log-softmax over the valid trie children only (mask-then-renormalize)
    Restricted modes only need the ~600 SID rows of the LM head.
    """

    tmpl = None

    def set_template(self, ids):
        """Cache the K/V of a prompt prefix shared by all requests (fused-first path only)."""
        self.tmpl = self.fwd.template_kv(ids, self.W.device)
        self.template_ids = list(ids)

    def _T0(self):
        return 0 if self.tmpl is None else self.tmpl[0].shape[2]

    level_w = (1.0, 1.0, 1.0)   # weights on the a, b, c terms (depth-discounted scoring)
    suffix_w = 1.0              # weight on lp(\n) + lp(EOS)

    def set_weights(self, wb=1.0, wc=1.0, ws=1.0):
        """Depth-discounted score: lp(a) + wb lp(b|a) + wc lp(c|ab) + ws (lp(\n)+lp(EOS)).
        All weights > 0 keep every term <= 0, so exact-search bounds stay admissible."""
        assert wb > 0 and wc > 0 and ws >= 0
        self.level_w = (1.0, float(wb), float(wc))
        self.suffix_w = float(ws)

    def _init_scoring(self, norm):
        assert norm in ('full', 'codebook', 'trie', 'cbtrie')   # cbtrie: codebook for a, b; trie for c
        self.norm = norm
        if norm != 'full':
            head = self.W.embed if isinstance(self.W.lm_head, torch.nn.Module) else self.W.lm_head
            self.sid_head = head[self.cat.sid_ids].contiguous()

    def _lp(self, h, level, tok_idx, valid):
        lp = self._lp_raw(h, level, tok_idx, valid)
        w = self.level_w[level]
        return lp if w == 1.0 else lp * w

    def _lp_raw(self, h, level, tok_idx, valid):
        if self.norm == 'full':
            lg = self.W.linear(h, self.W.lm_head)[..., :self.W.V].float()
            return lg.gather(-1, tok_idx) - torch.logsumexp(lg, -1, keepdim=True)
        lg = torch.nn.functional.linear(h, self.sid_head).float()
        g = lg.gather(-1, self.cat.tok2loc[tok_idx])
        if self.norm == 'codebook' or (self.norm == 'cbtrie' and level < 2):
            lse = torch.logsumexp(lg.masked_fill(~self.cat.level_mask[level], float('-inf')), -1, keepdim=True)
        else:
            lse = torch.logsumexp(g if valid is None else g.masked_fill(~valid, float('-inf')), -1, keepdim=True)
        return g - lse


class BeamEngine(ScoreMixin):
    def __init__(self, W, cat, beam=50, newline_id=198, eos_id=151645, suffix_mode='fused', norm='full',
                 fuse_first=False):
        self.W, self.cat, self.beam = W, cat, beam
        self.fuse_first = fuse_first
        self.fwd = Forward(W)
        self.newline, self.eos = newline_id, eos_id
        assert suffix_mode in ('fused', 'separate', 'none')
        self._init_scoring(norm)
        self.suffix_mode = suffix_mode if norm == 'full' else 'none'

    def alloc(self, B, P):
        W, dev, dt = self.W, self.W.device, self.W.dtype
        R = self.beam
        return dict(
            kc=torch.zeros(W.L, B, W.KV, P, W.hd, device=dev, dtype=dt),
            vc=torch.zeros(W.L, B, W.KV, P, W.hd, device=dev, dtype=dt),
            kp=torch.zeros(W.L, B, R, 4, W.KV, W.hd, device=dev, dtype=dt),
            vp=torch.zeros(W.L, B, R, 4, W.KV, W.hd, device=dev, dtype=dt),
            kn=torch.zeros(W.L, B, len(self.cat.first), 1, W.KV, W.hd, device=dev, dtype=dt),
            vn=torch.zeros(W.L, B, len(self.cat.first), 1, W.KV, W.hd, device=dev, dtype=dt),
        )

    def _logits(self, h):
        return self.W.linear(h, self.W.lm_head)[..., :self.W.V].float()

    @staticmethod
    def _first_topk(la, R):
        # catalogs with fewer valid first codes than beams: extra beams are dead (-inf), as in HF
        k = min(R, la.shape[-1])
        s1, ai = la.topk(k, -1)
        if k < R:
            s1 = torch.cat([s1, s1.new_full((s1.shape[0], R - k), float('-inf'))], -1)
            ai = torch.cat([ai, ai.new_zeros((ai.shape[0], R - k))], -1)
        return s1, ai

    @staticmethod
    def _reorder(buf, bidx, par):
        buf.copy_(buf[:, bidx, par])

    def run(self, tokens, lengths, bufs):
        W, cat, R, fwd = self.W, self.cat, self.beam, self.fwd
        B, P = tokens.shape
        dev = tokens.device
        kc, vc, kp, vp = bufs['kc'], bufs['vc'], bufs['kp'], bufs['vp']
        bidx = torch.arange(B, device=dev)[:, None]
        neg = torch.finfo(torch.float32).min
        pmask = torch.where(torch.arange(P, device=dev)[None, :] < lengths[:, None], 0.0, neg)
        L0 = lengths[:, None, None]
        nA = len(cat.first)
        if self.fuse_first:
            # ---- prompt + every first-level trie node in one forward
            T0 = self._T0()
            hp, hn = fwd.prefill_tree(W.embed[tokens[:, T0:]], W.embed[cat.first][None].expand(B, nA, W.D), lengths,
                                      kc, vc, bufs['kn'], bufs['vn'], tmpl=self.tmpl)
            la = self._lp(hp[torch.arange(B, device=dev), lengths - 1 - T0], 0, cat.first[None].expand(B, -1), None)
            s1, ai = self._first_topk(la, R)  # [B,R]
            a_tok = cat.first[ai]
            t1 = cat.t1[ai]; v1 = cat.v1[ai]
            lb = self._lp(hn.gather(1, ai[..., None].expand(B, R, W.D)), 1, t1, v1)
            kp[:, :, :, 0] = bufs['kn'][:, bidx, ai, 0]
            vp[:, :, :, 0] = bufs['vn'][:, bidx, ai, 0]
        else:
            # ---- prefill (one row per user)
            pos = torch.arange(P, device=dev)[None, :].expand(B, P)
            h = fwd.prefill(W.embed[tokens], pos, kc, vc)
            hl = h[torch.arange(B, device=dev), lengths - 1]
            la = self._lp(hl, 0, cat.first[None].expand(B, -1), None)
            s1, ai = self._first_topk(la, R)  # [B,R]
            a_tok = cat.first[ai]
            # ---- step b
            h = fwd.step(W.embed[a_tok][:, :, None], L0.expand(B, R, 1), kc, vc, pmask, kp, vp, 0, lengths=lengths)
            t1 = cat.t1[ai]; v1 = cat.v1[ai]
            lb = self._lp(h[:, :, 0], 1, t1, v1)
        cand = (s1[..., None] + lb).masked_fill(~v1, float('-inf')).view(B, R * cat.W1)
        s2, idx = cand.topk(R, -1)
        par = idx // cat.W1
        col = idx % cat.W1
        b_tok = t1.view(B, -1).gather(1, idx)
        pair = cat.pair_table[ai.gather(1, par), col]
        tr_a = s1.gather(1, par)
        tr_b = lb.view(B, -1).gather(1, idx)
        a_tok = a_tok.gather(1, par)
        self._reorder(kp, bidx, par); self._reorder(vp, bidx, par)
        # ---- step c
        h = fwd.step(W.embed[b_tok][:, :, None], (L0 + 1).expand(B, R, 1), kc, vc, pmask, kp, vp, 1, lengths=lengths)
        t2 = cat.t2[pair]; v2 = cat.v2[pair]
        lc = self._lp(h[:, :, 0], 2, t2, v2)
        cand = (s2[..., None] + lc).masked_fill(~v2, float('-inf')).view(B, R * cat.W2)
        s3, idx = cand.topk(R, -1)
        par = idx // cat.W2
        c_tok = t2.view(B, -1).gather(1, idx)
        tr_c = lc.view(B, -1).gather(1, idx)
        a_tok, b_tok = a_tok.gather(1, par), b_tok.gather(1, par)
        tr_a, tr_b = tr_a.gather(1, par), tr_b.gather(1, par)
        sids = torch.stack([a_tok, b_tok, c_tok], -1)
        if self.suffix_mode == 'none':
            final = s3
            tr = torch.stack([tr_a, tr_b, tr_c], -1)
        else:
            self._reorder(kp, bidx, par); self._reorder(vp, bidx, par)
            nl = torch.full_like(c_tok, self.newline)
            if self.suffix_mode == 'fused':
                x = W.embed[torch.stack([c_tok, nl], -1)]  # [B,R,2,D]
                pos = torch.cat([L0 + 2, L0 + 3], -1).expand(B, R, 2)
                h = fwd.step(x, pos, kc, vc, pmask, kp, vp, 2, lengths=lengths)
                lg = self._logits(h)  # [B,R,2,V]
                lse = torch.logsumexp(lg, -1)
                tr_n = lg[..., 0, self.newline] - lse[..., 0]
                tr_e = lg[..., 1, self.eos] - lse[..., 1]
            else:
                h = fwd.step(W.embed[c_tok][:, :, None], (L0 + 2).expand(B, R, 1), kc, vc, pmask, kp, vp, 2, lengths=lengths)
                lg = self._logits(h[:, :, 0])
                tr_n = lg[..., self.newline] - torch.logsumexp(lg, -1)
                h = fwd.step(W.embed[nl][:, :, None], (L0 + 3).expand(B, R, 1), kc, vc, pmask, kp, vp, 3, lengths=lengths)
                lg = self._logits(h[:, :, 0])
                tr_e = lg[..., self.eos] - torch.logsumexp(lg, -1)
            if self.suffix_w != 1.0:
                tr_n, tr_e = tr_n * self.suffix_w, tr_e * self.suffix_w
            final = s3 + tr_n + tr_e
            tr = torch.stack([tr_a, tr_b, tr_c, tr_n, tr_e], -1)
        final, order = torch.sort(final, dim=-1, descending=True, stable=True)
        return {'sids': sids.gather(1, order[..., None].expand(B, R, 3)),
                'scores': final,
                'trans': tr.gather(1, order[..., None].expand(B, R, tr.shape[-1]))}


_POOL = None


class GraphRunner:
    """Captures engine.run for one (B, P) bucket as a single CUDA graph.

    All graphs share one memory pool: requests are replayed sequentially, so
    intermediate buffers of different buckets may alias.
    """

    def __init__(self, engine, B, P, pad_id, warmup=3, use_graph=True):
        global _POOL
        self.engine, self.B, self.P = engine, B, P
        dev = engine.W.device
        self.tokens = torch.full((B, P), pad_id, dtype=torch.long, device=dev)
        self.lengths = torch.full((B,), P, dtype=torch.long, device=dev)
        self.bufs = engine.alloc(B, P)
        self.pad_id = pad_id
        self.graph = None
        if dev.type == 'cuda':
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    self.out = engine.run(self.tokens, self.lengths, self.bufs)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            if use_graph:
                if _POOL is None:
                    _POOL = torch.cuda.graph_pool_handle()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, pool=_POOL, capture_error_mode="relaxed"):
                    self.out = engine.run(self.tokens, self.lengths, self.bufs)
                torch.cuda.synchronize()

    def load(self, token_lists):
        """Host-side staging (outside the timed region): pad and copy prompts."""
        B, P = self.B, self.P
        t = torch.full((B, P), self.pad_id, dtype=torch.long)
        ln = torch.full((B,), 1, dtype=torch.long)
        for i, ids in enumerate(token_lists):
            t[i, :len(ids)] = torch.tensor(ids)
            ln[i] = len(ids)
        return t.pin_memory() if self.tokens.is_cuda else t, ln.pin_memory() if self.tokens.is_cuda else ln

    def __call__(self, t, ln):
        self.tokens.copy_(t, non_blocking=True)
        self.lengths.copy_(ln, non_blocking=True)
        if self.graph is not None:
            self.graph.replay()
            return self.out
        return self.engine.run(self.tokens, self.lengths, self.bufs)
