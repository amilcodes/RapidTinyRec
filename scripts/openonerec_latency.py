"""Latency of certified exact top-K vs beam search on OpenOneRec-1.7B, in our engine (Qwen3 port).

One engine runs both decoders. A per-level budget (A1, M) keeps the A1 best first codes by s(a), expands them,
keeps the M best prefixes by U2 = s(a) + s(b|a), expands those, and returns the top-K items. With A1 = M = B this
is exactly trie-constrained beam search of width B. The same forward also yields the certificate: the returned
top-K is the exact top-K of the objective whenever its K-th score is at least the best bound left outside the
budget, max(s(a) of the first unexpanded first code, U2 of the first unexpanded prefix). Every config therefore
reports its latency and whether its answer is provably exact, on the same model, prompts, catalog, objective
(full-vocabulary log-softmax over valid SIDs, as in openonerec_cert.py) and GPU.

Batch 1, BF16 weights and activations, FP32 logits. The prompt forward is one CUDA graph per request (captured at
its exact length; it is the same for every decoder); the two decode forwards plus all selection, scoring and
certification are one CUDA graph per budget. Prompt attention for the many first-level rows uses
kernels.prefix_attention (each prompt chunk is read once for a block of rows).

usage: python scripts/openonerec_latency.py --model M --data D.parquet --catalog C --out R.jsonl \
           [--users 200] [--configs 32/32,128/128,...] [--ref openonerec_cert_output.jsonl]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from openonerec_cert import load_catalog, parse_messages, targets_of  # noqa: E402
from rapidtinyrec.speedrun.kernels import merge_attention, prefix_attention, prefix_plan, selfcheck_prefix  # noqa: E402
from rapidtinyrec.speedrun.model import Forward, Qwen2Weights  # noqa: E402

NEG = float('-inf')


class OOEngine:
    def __init__(self, W, ids, keys, pmax, kmax=32):
        self.W, self.fwd, self.kmax = W, Forward(W), kmax
        dev = W.device
        self.ida, self.idb, self.idc = (torch.tensor(ids[lv], device=dev) for lv in 'abc')
        self.keys = torch.from_numpy(keys).to(dev)                    # sorted packed (a<<26 | b<<13 | c)
        first = np.unique(keys >> 26)
        self.n_first = len(first)
        va = torch.zeros(8192, dtype=torch.bool)
        va[torch.from_numpy(first)] = True
        self.valid_a = va.to(dev)
        pk, cnt = np.unique(keys >> 13, return_counts=True)
        vb = torch.zeros(8192 * 8192, dtype=torch.bool)
        vb[torch.from_numpy(pk)] = True
        self.valid_b = vb.view(8192, 8192).to(dev)                   # valid (a, b) prefixes
        self.n_prefix, self.Cmax = len(pk), int(cnt.max())
        self.car = torch.arange(self.Cmax, device=dev)
        self.scale = 1.0 / math.sqrt(W.hd)
        self.pmax = pmax
        self.kc = torch.zeros(W.L, W.KV, pmax, W.hd, device=dev, dtype=W.dtype)
        self.vc = torch.zeros_like(self.kc)
        self.length = torch.zeros(1, device=dev, dtype=torch.int32)
        self.lpa = torch.full((8192,), NEG, device=dev)
        self.P = torch.zeros((), device=dev, dtype=torch.long)
        # the layer's small ops (RMSNorms, QK-norm, RoPE, SiLU, residuals, the private-slot attention and the
        # merge) are fused by torch.compile; GEMMs stay cuBLAS. Without this the Qwen3 layer is launch-bound.
        torch._dynamo.config.cache_size_limit = 64
        self.qkv = torch.compile(self.fwd.qkv, dynamic=True)
        self.post = torch.compile(self.fwd.post, dynamic=True)
        self.priv = torch.compile(self._private_merge, dynamic=True)
        self.lse_cols = torch.compile(self._lse_cols, dynamic=True)

    # ---- prefill: prompt K/V into the cache, s(a) for every valid first code
    def prefill(self, ids):
        W = self.W
        P = ids.shape[1]
        assert P + 2 <= self.pmax
        rep = W.H // W.KV
        x = W.embed[ids[0]]
        pos = torch.arange(P, device=ids.device)
        for i, layer in enumerate(W.layers):
            q, k, v = self.qkv(x, layer, pos)                         # [P,H,hd], [P,KV,hd]
            kt, vt = k.transpose(0, 1), v.transpose(0, 1)
            self.kc[i][:, :P] = kt
            self.vc[i][:, :P] = vt
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kt.repeat_interleave(rep, 0)[None],
                                               vt.repeat_interleave(rep, 0)[None], is_causal=True, scale=self.scale)
            x = self.post(x, o[0].transpose(0, 1).reshape(P, W.H * W.hd), layer)
        lpa = self.level_lp(self.fwd.final(x[-1:]), self.ida)[0]
        self.lpa.copy_(torch.where(self.valid_a, lpa, NEG))
        self.length.fill_(P)
        self.P.fill_(P)

    def _private_merge(self, q, ks, vs, o, lse, rep):
        """attention to the rows' own slots ks/vs [R,S,KV,hd], merged with the prompt chunks (o, lse)."""
        ks = ks.repeat_interleave(rep, 2).float()
        vs = vs.repeat_interleave(rep, 2).float()
        ss = (q.float()[:, None] * ks).sum(-1).transpose(1, 2) * self.scale  # [R,H,S]
        os_ = (torch.softmax(ss, -1).transpose(1, 2)[..., None] * vs).sum(1)  # [R,H,hd]
        return merge_attention(o, lse, os_, torch.logsumexp(ss, -1)).to(q.dtype).flatten(1)

    def _lse_cols(self, lg, ids):
        return lg[:, ids] - torch.logsumexp(lg, 1, keepdim=True)

    def level_lp(self, h, ids):
        """log-probabilities of the level's 8,192 codes under the full-vocabulary softmax. The LM head writes FP32
        logits from BF16 inputs: rounding logits of magnitude ~30 to BF16 (steps of 0.125) creates spurious ties."""
        lg = torch.mm(h, self.W.lm_head.t(), out_dtype=torch.float32)[:, :self.W.V]
        return self.lse_cols(lg, ids)

    # ---- one decode forward for R rows of one request; private slots [0, s0] (slot s0 = this token)
    def decode(self, x, pos, kp, vp, s0, plan):
        W = self.W
        rep = W.H // W.KV
        for i, layer in enumerate(W.layers):
            q, k, v = self.qkv(x, layer, pos)                         # q [R,H,hd], k/v [R,KV,hd]
            kp[i][:, s0] = k
            vp[i][:, s0] = v
            o, lse = prefix_attention(q, self.kc[i], self.vc[i], self.length, self.scale, plan)
            x = self.post(x, self.priv(q, kp[i][:, :s0 + 1], vp[i][:, :s0 + 1], o, lse, rep), layer)
        return self.fwd.final(x)

    # ---- budgeted search; A1 = M = B is beam search
    def search(self, A1, M, bufs):
        W, dev = self.W, self.W.device
        kp2, vp2, kp3, vp3, plan2, plan3 = bufs
        va, ia = torch.topk(self.lpa, A1 + 1)
        asel, b1 = ia[:A1], va[A1]                                    # b1: best s(a) left out
        pos2 = self.P.expand(A1)
        h2 = self.decode(W.embed[self.ida[asel]], pos2, kp2, vp2, 0, plan2)
        u2 = self.lpa[asel][:, None] + self.level_lp(h2, self.idb)  # [A1, 8192]
        u2 = torch.where(self.valid_b[asel], u2, NEG)
        vu, iu = torch.topk(u2.view(-1), M + 1)
        b2 = vu[M]                                                    # best U2 left out
        r, bs = iu[:M] // 8192, iu[:M] % 8192
        kp3[:, :, 0] = kp2[:, r, 0]
        vp3[:, :, 0] = vp2[:, r, 0]
        h3 = self.decode(W.embed[self.idb[bs]], (self.P + 1).expand(M), kp3, vp3, 1, plan3)
        lpc = self.level_lp(h3, self.idc)                            # [M, 8192]
        pre = (asel[r] << 13) | bs
        st = torch.searchsorted(self.keys, pre << 13)
        en = torch.searchsorted(self.keys, (pre + 1) << 13)
        j = st[:, None] + self.car[None]
        ok = j < en[:, None]
        cc = self.keys[j.clamp(max=self.keys.numel() - 1)] & 8191
        sc = torch.where(ok, vu[:M, None] + lpc.gather(1, cc), NEG)
        vt, it = torch.topk(sc.view(-1), min(self.kmax, M * self.Cmax))
        row = it // self.Cmax
        items = torch.stack([asel[r[row]], bs[row], cc.view(-1)[it]], 1)
        return vt, items, torch.maximum(b1, b2), b1, b2

    def graph(self, A1, M):
        W, dev = self.W, self.W.device
        A1 = min(A1, self.n_first)
        rep = W.H // W.KV
        kp2 = torch.zeros(W.L, A1, 2, W.KV, W.hd, device=dev, dtype=W.dtype)
        vp2 = torch.zeros_like(kp2)
        kp3 = torch.zeros(W.L, M, 2, W.KV, W.hd, device=dev, dtype=W.dtype)
        vp3 = torch.zeros_like(kp3)
        bufs = (kp2, vp2, kp3, vp3, prefix_plan(A1, self.pmax, W.KV, rep), prefix_plan(M, self.pmax, W.KV, rep))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.search(A1, M, bufs)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.search(A1, M, bufs)
        return dict(g=g, out=out, A1=A1, M=M, eager=lambda: self.search(A1, M, bufs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--users', type=int, default=200, help='same sampling as openonerec_cert.py (seed 0)')
    ap.add_argument('--configs', default='32/32,128/128,64/16,128/32,256/64,512/64,1024/256,2048/256,4096/512')
    ap.add_argument('--ladders', default='32/32;128/128;128/32>512/64>2048/256>4096/512;'
                                        '256/64>1024/256>4096/512;512/64>2048/256>4096/512',
                    help='policies timed wall-clock: budgets tried in order until the top-K certifies (host check '
                         'after each); a single budget is a plain decoder')
    ap.add_argument('--ladder-k', default='10,32')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--ladder-reps', type=int, default=3)
    ap.add_argument('--ref', default='', help='openonerec_cert.py output for the same task (exact top-32 to match)')
    ap.add_argument('--check', type=int, default=3, help='requests whose top items are re-scored with HF Transformers')
    ap.add_argument('--target-field', default='metadata')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import pandas as pd
    from transformers import AutoTokenizer
    dev = torch.device('cuda')
    print(json.dumps({'stage': 'gpu', 'name': torch.cuda.get_device_name(0),
                      'prefix_kernel_maxabs': selfcheck_prefix()}), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = {lv: np.array(tok.convert_tokens_to_ids([f'<s_{lv}_{i}>' for i in range(8192)]), dtype=np.int64)
           for lv in 'abc'}
    keys = load_catalog(a.catalog)
    df = pd.read_parquet(a.data)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(df), size=min(a.users, len(df)), replace=False)
    reqs = []
    for ri in rows:
        row = df.iloc[int(ri)].to_dict()
        msgs = parse_messages(row['messages'])
        pm = [m for m in msgs if m.get('role') != 'assistant']
        targets = targets_of({'answer': [m for m in msgs if m.get('role') == 'assistant']})
        if not targets and a.target_field:
            targets = targets_of({'t': row.get(a.target_field)})
        text = tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        reqs.append((int(ri), tok(text + '<|sid_begin|>').input_ids, sorted(targets)))
    pmax = max(len(r[1]) for r in reqs) + 64
    t0 = time.time()
    W = Qwen2Weights(a.model, torch.bfloat16, dev)
    E = OOEngine(W, ids, keys, pmax)
    cfgs = [tuple(int(x) for x in c.split('/')) for c in a.configs.split(',')]
    G = {f'{A1}/{M}': E.graph(A1, M) for A1, M in cfgs}
    print(json.dumps({'stage': 'setup', 'seconds': time.time() - t0, 'n_sids': int(len(keys)), 'n_first': E.n_first,
                      'n_prefix': E.n_prefix, 'max_children_2': E.Cmax, 'pmax': pmax,
                      'graphs': {k: [v['A1'], v['M']] for k, v in G.items()},
                      'mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30}), flush=True)
    ref = {}
    if a.ref and os.path.exists(a.ref):
        for l in open(a.ref):
            r = json.loads(l)
            ref[r['row']] = r['full']
    hf = None
    if a.check:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(dev).eval()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ids_buf = torch.zeros(pmax, device=dev, dtype=torch.long)
    f = open(a.out, 'w')
    for n, (ri, tids, targets) in enumerate(reqs):
        # prefill: one CUDA graph at this request's exact length (captured per request, timed over replays)
        P = len(tids)
        ids_buf[:P].copy_(torch.tensor(tids, device=dev))
        x = ids_buf[None, :P]
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            E.prefill(x)
        torch.cuda.current_stream().wait_stream(s_)
        gp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gp):
            E.prefill(x)
        pts = []
        for _ in range(a.reps + 1):
            ev[0].record()
            gp.replay()
            ev[1].record()
            torch.cuda.synchronize()
            pts.append(ev[0].elapsed_time(ev[1]))
        del gp
        rec = dict(row=ri, prompt_tokens=P, targets=targets, prefill_ms=float(np.median(pts[1:])), cfg={})
        for name, gr in G.items():
            gr['g'].replay()
            ts = []
            for _ in range(a.reps):
                ev[0].record()
                gr['g'].replay()
                ev[1].record()
                torch.cuda.synchronize()
                ts.append(ev[0].elapsed_time(ev[1]))
            vt, items, bound, b1, b2 = (t.clone() for t in gr['out'])
            vt = vt.tolist()
            bound = float(bound)
            rec['cfg'][name] = dict(ms=float(np.median(ts)), ms_min=float(min(ts)),
                                    **{f'cert@{k}': bool(k <= len(vt) and vt[k - 1] >= bound) for k in (1, 10, 32)},
                                    bound=bound, bound_a=float(b1), bound_p=float(b2),
                                    top32=items.tolist(), scores32=vt)
        rec['ladder'] = {}
        for K in (int(k) for k in a.ladder_k.split(',')):
            for spec in a.ladders.split(';'):
                rungs = spec.split('>')
                ts = []
                for _ in range(a.ladder_reps + 1):
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    for j, nm in enumerate(rungs):
                        G[nm]['g'].replay()
                        vt_, bd_ = G[nm]['out'][0], G[nm]['out'][2]
                        ok = K <= vt_.numel() and bool((vt_[K - 1] >= bd_).item())   # host sync: certificate
                        if ok:
                            break
                    ts.append((time.perf_counter() - t1) * 1e3)
                rec['ladder'][f'{spec}@{K}'] = dict(ms=float(np.median(ts[1:])), rung=j, cert=ok)
        big = list(G)[-1]
        if ri in ref:                                                 # agreement with the HF reference search
            mine, theirs = rec['cfg'][big], ref[ri]
            rec['ref_top10_overlap'] = len({tuple(t) for t in mine['top32'][:10]} & {tuple(t) for t in theirs['top32'][:10]}) / 10
            rs = {tuple(t): s for t, s in zip(theirs['top32'], theirs['scores32'])}
            d = [abs(s - rs[tuple(t)]) for t, s in zip(mine['top32'], mine['scores32']) if tuple(t) in rs]
            rec['ref_score_maxabs'] = max(d) if d else None
        if hf is not None and n < a.check:                            # re-score our top-3 sequentially in HF
            errs = []
            for (ia, ib, ic), s in list(zip(rec['cfg'][big]['top32'], rec['cfg'][big]['scores32']))[:3]:
                seq = torch.tensor([tids + [int(ids['a'][ia]), int(ids['b'][ib])]], device=dev)
                lg = hf(input_ids=seq).logits[0, -3:].float().log_softmax(-1)
                hs = float(lg[0, ids['a'][ia]] + lg[1, ids['b'][ib]] + lg[2, ids['c'][ic]])
                errs.append(abs(hs - s))
            rec['hf_rescore_maxabs'] = max(errs)
        f.write(json.dumps(rec) + '\n')
        f.flush()
        if n % 20 == 0 or n < 3:
            print(json.dumps({'stage': 'req', 'n': n, 'P': len(tids), 'prefill_ms': round(rec['prefill_ms'], 2),
                              **{k: [round(v['ms'], 2), v['cert@10']] for k, v in rec['cfg'].items()},
                              'ref_overlap': rec.get('ref_top10_overlap'), 'hf_err': rec.get('hf_rescore_maxabs')}),
                  flush=True)
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
