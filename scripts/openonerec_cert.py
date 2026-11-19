"""How much does certified exact decoding need on a real wide first level? OpenOneRec (Qwen3, 3 x 8,192 SIDs).

For each benchmark request we run certified best-first search over the catalog trie with Hugging Face transformers and
tree attention (candidate rows attend to the cached prompt and to their own ancestors), and record:
  * the instance-optimal expansion (Prop. 2 of the paper): first codes with s(a) > tau*_K and prefixes with
    U2(a,b) > tau*_K, where tau*_K is the exact K-th item score, for K in --ks;
  * whether beam search (width B) returns the exact top-K, and the target hit rate of both;
under three scoring rules computed from the same logits:
  full      log-softmax over the whole vocabulary, invalid items masked, no renormalization (STATIC, OpenOneRec eval)
  codebook  log-softmax over the 8,192 tokens of the current level (GR4AD-style)
  trie      log-softmax over the valid children of the current prefix (xGR-style mask-before-softmax)
Every term is a log-probability <= 0, so s(a) bounds every item under a and U2 = s(a) + s(b|a) every item under (a,b).

usage: python openonerec_cert.py --model DIR --data video_test.parquet --catalog sid2pid.json --out OUT.jsonl
       [--users 200] [--ks 1,10,32,128,512] [--beams 32,128] [--norms full,codebook,trie]
"""
import argparse
import heapq
import json
import math
import re
import sys
import time

import numpy as np
import torch

SID_RE = re.compile(r's_a_(\d+)>\s*<s_b_(\d+)>\s*<s_c_(\d+)')


def pack(a, b, c):
    return (int(a) << 26) | (int(b) << 13) | int(c)       # = (a * 8192 + b) * 8192 + c, RecIF's sid2pid key format


def load_catalog(path):
    """Valid SIDs as packed int64 keys. Accepts RecIF's sid2pid/sid2iid JSON (packed integer keys), any JSON/parquet
    holding <s_a_*><s_b_*><s_c_*> strings, or a pid2sid parquet with an array column `sid` = [a, b, c]."""
    if path.endswith('.json'):
        d = json.load(open(path))
        keys = []
        for k in (d.keys() if isinstance(d, dict) else d):
            m = SID_RE.search(str(k))
            if m:
                keys.append(pack(*m.groups()))
                continue
            parts = re.findall(r'\d+', str(k))
            if len(parts) == 3:
                keys.append(pack(*parts))
            elif len(parts) == 1:
                keys.append(int(parts[0]))
        return np.unique(np.array(keys, dtype=np.int64))
    import pandas as pd
    df = pd.read_parquet(path)
    if 'sid' in df.columns and not isinstance(df['sid'].iloc[0], str):
        s = np.stack(df['sid'].values).astype(np.int64)
        return np.unique((s[:, 0] << 26) | (s[:, 1] << 13) | s[:, 2])
    keys = set()
    for col in df.columns:
        hits = df[col].astype(str).str.extract(r's_a_(\d+)>\s*<s_b_(\d+)>\s*<s_c_(\d+)').dropna()
        keys |= {pack(*r) for r in hits.values.tolist()}
    return np.unique(np.array(sorted(keys), dtype=np.int64))


class Trie:
    """Catalog trie over packed SID keys, built with sorted-array splits (fast for the 14.6M-SID corpus)."""

    def __init__(self, keys):
        keys = np.unique(keys)
        ab, c = keys >> 13, keys & 8191
        uab, s2 = np.unique(ab, return_index=True)
        e2 = np.r_[s2[1:], len(keys)]
        self.ch2 = {(int(x >> 13), int(x & 8191)): c[i:j] for x, i, j in zip(uab.tolist(), s2.tolist(), e2.tolist())}
        ua, s1 = np.unique(uab >> 13, return_index=True)
        e1 = np.r_[s1[1:], len(uab)]
        b = uab & 8191
        self.ch1 = {int(x): b[i:j] for x, i, j in zip(ua.tolist(), s1.tolist(), e1.tolist())}
        self.first = ua.astype(np.int64)
        self.n_sids = len(keys)
        self.keys = keys

    def contains(self, sid):
        k = pack(*sid)
        i = np.searchsorted(self.keys, k)
        return bool(i < len(self.keys) and self.keys[i] == k)


def compact(logits, lvl_ids):
    """logits [R, V] -> per row (lse over the vocabulary, lse over the level's 8,192 codes, level logits [8192])."""
    lse_full = torch.logsumexp(logits, -1).cpu().numpy()
    lvl = logits[:, lvl_ids]
    lse_cb = torch.logsumexp(lvl, -1).cpu().numpy()
    lvl = lvl.cpu().numpy()
    return [(float(lse_full[r]), float(lse_cb[r]), lvl[r]) for r in range(logits.shape[0])]


def lp(node, valid, norm):
    """Log-probabilities of the valid child codes of one expanded node under a scoring rule."""
    lse_full, lse_cb, lvl = node
    g = lvl[valid]
    if norm == 'full':
        return g - lse_full
    if norm == 'codebook':
        return g - lse_cb
    m = g.max()
    return g - (m + np.log(np.exp(g - m).sum()))


class Searcher:
    """Tree-attention expansions over a cached prompt, with per-request caches of node logits."""

    def __init__(self, model, tok_ids, dev, dtype):
        self.model, self.dev, self.dtype = model, dev, dtype
        self.ids = tok_ids                                     # dict level -> np.array[8192] token ids

    @torch.inference_mode()
    def prompt(self, input_ids):
        from transformers import DynamicCache
        self.cache = DynamicCache()
        out = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True)
        self.L = input_ids.shape[1]
        self.root = compact(out.logits[0, -1:].float(), self.ids['a'])[0]

    @torch.inference_mode()
    def expand(self, paths, chunk):
        """paths: list of tuples of codes ((a,) or (a, b)) whose next-level logits are needed. Each path becomes a
        chain of rows after the prompt (tree attention); returns compact next-level scores for each path."""
        res = []
        for s in range(0, len(paths), chunk):
            part = paths[s:s + chunk]
            depth = len(part[0])
            toks, pos, owner = [], [], []
            for j, p in enumerate(part):
                for d, code in enumerate(p):
                    toks.append(int(self.ids['abc'[d]][code]))
                    pos.append(self.L + d)
                    owner.append((j, d))
            n = len(toks)
            neg = torch.finfo(self.dtype).min
            mask = torch.full((n, self.L + n), neg, dtype=self.dtype, device=self.dev)
            mask[:, :self.L] = 0
            for i, (j, d) in enumerate(owner):
                for d2 in range(d + 1):                       # ancestors in the same chain and itself
                    mask[i, self.L + i - (d - d2)] = 0
            out = self.model(input_ids=torch.tensor([toks], device=self.dev),
                             position_ids=torch.tensor([pos], device=self.dev),
                             attention_mask=mask[None, None], past_key_values=self.cache, use_cache=True)
            self.cache.crop(self.L)
            ends = [i for i, (j, d) in enumerate(owner) if d == depth - 1]
            res += compact(out.logits[0, ends].float(), self.ids['abc'[depth]])
        return res


def exact_search(S, trie, norm, kmax, chunk_a, chunk_p, max_a, max_p, cache_lp):
    """Certified best-first search for the top-kmax items (stops early at the expansion caps). Returns the best kmax
    exact scores, the final unexpanded bound, and the bounds needed for instance-optimal expansion counts."""
    la = lp(S.root, trie.first, norm)                        # s(a), valid first codes
    order = np.argsort(-la)
    s1 = dict(zip(trie.first.tolist(), la.tolist()))
    u2 = {}                                                   # (a,b) -> U2 for expanded first codes
    pheap = []                                                # (-U2, a, b) unexpanded prefixes
    pool = []                                                 # min-heap of the best kmax exact scores
    ia, nA = 0, len(order)
    exp_a = exp_p = 0
    while True:
        tauA = la[order[ia]] if ia < nA else -math.inf
        tauP = -pheap[0][0] if pheap else -math.inf
        kth = pool[0][0] if len(pool) >= kmax else -math.inf
        if kth >= max(tauA, tauP):
            break
        if tauA >= tauP and exp_a < max_a:
            batch = [int(trie.first[order[k]]) for k in range(ia, min(nA, ia + chunk_a))]
            ia += len(batch)
            exp_a += len(batch)
            need = [(a,) for a in batch if ('a', a) not in cache_lp]
            for (a,), node in zip(need, S.expand(need, chunk_a) if need else []):
                cache_lp[('a', a)] = node
            for a in batch:
                v = trie.ch1[a]
                for b, x in zip(v.tolist(), lp(cache_lp[('a', a)], v, norm).tolist()):
                    u = s1[a] + x
                    u2[(a, b)] = u
                    heapq.heappush(pheap, (-u, a, b))
        elif pheap and exp_p < max_p:
            batch = []
            while pheap and len(batch) < chunk_p:
                _, a, b = heapq.heappop(pheap)
                batch.append((a, b))
            exp_p += len(batch)
            need = [p for p in batch if ('ab',) + p not in cache_lp]
            for p, node in zip(need, S.expand(need, chunk_p) if need else []):
                cache_lp[('ab',) + p] = node
            for a, b in batch:
                v = trie.ch2[(a, b)]
                for c, x in zip(v.tolist(), lp(cache_lp[('ab', a, b)], v, norm).tolist()):
                    sc = u2[(a, b)] + x
                    if len(pool) < kmax:
                        heapq.heappush(pool, (sc, (a, b, c)))
                    elif sc > pool[0][0]:
                        heapq.heapreplace(pool, (sc, (a, b, c)))
        else:
            break
    tauA = la[order[ia]] if ia < nA else -math.inf
    tauP = -pheap[0][0] if pheap else -math.inf
    top = sorted(pool, key=lambda t: -t[0])
    return dict(top=top, bound=max(tauA, tauP), s1=s1, u2=u2, exp_a=exp_a, exp_p=exp_p)


def beam_search(S, trie, norm, B, cache_lp, chunk):
    la = lp(S.root, trie.first, norm)
    top_a = trie.first[np.argsort(-la)[:B]].tolist()
    s1 = dict(zip(trie.first.tolist(), la.tolist()))
    need = [(a,) for a in top_a if ('a', a) not in cache_lp]
    for (a,), node in zip(need, S.expand(need, chunk) if need else []):
        cache_lp[('a', a)] = node
    cand = []
    for a in top_a:
        v = trie.ch1[a]
        cand += [(s1[a] + x, a, b) for b, x in zip(v.tolist(), lp(cache_lp[('a', a)], v, norm).tolist())]
    cand = heapq.nlargest(B, cand)
    need = [(a, b) for _, a, b in cand if ('ab', a, b) not in cache_lp]
    for p, node in zip(need, S.expand(need, chunk) if need else []):
        cache_lp[('ab',) + p] = node
    fin = []
    for u, a, b in cand:
        v = trie.ch2[(a, b)]
        fin += [(u + x, (a, b, c)) for c, x in zip(v.tolist(), lp(cache_lp[('ab', a, b)], v, norm).tolist())]
    return heapq.nlargest(B, fin)


def targets_of(row):
    """Ground-truth SIDs of a benchmark row (searched in the answer message and the metadata)."""
    blob = json.dumps({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in row.items()}, default=str)
    return {tuple(int(x) for x in m) for m in SID_RE.findall(blob)}


def parse_messages(msgs):
    """Chat messages as [{'role', 'content': str}]. RecIF stores them as a JSON string whose contents are lists of
    {'type': 'text', 'text': ...} parts; Qwen3's chat template drops non-string content, so the parts are joined."""
    if isinstance(msgs, (str, bytes)):
        msgs = json.loads(msgs)
    out = []
    for m in (msgs.tolist() if hasattr(msgs, 'tolist') else msgs):
        m = dict(m)
        c = m.get('content')
        if not isinstance(c, str):
            parts = c.tolist() if hasattr(c, 'tolist') else (c or [])
            c = ''.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in parts)
        out.append({'role': m['role'], 'content': c})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--catalog', required=True, help='json/parquet holding every valid SID')
    ap.add_argument('--out', required=True)
    ap.add_argument('--users', type=int, default=200)
    ap.add_argument('--ks', default='1,10,32,128,512')
    ap.add_argument('--beams', default='32,128')
    ap.add_argument('--norms', default='full,codebook,trie')
    ap.add_argument('--chunk-a', type=int, default=128)
    ap.add_argument('--chunk-p', type=int, default=512)
    ap.add_argument('--max-a', type=int, default=8192)
    ap.add_argument('--max-p', type=int, default=40000)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--target-field', default='', help='column holding the ground truth if not in an assistant message')
    ap.add_argument('--check', type=int, default=2, help='requests whose tree-attention logits are checked sequentially')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device('cuda')
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, attn_implementation='sdpa').to(dev).eval()
    ids = {lv: np.array(tok.convert_tokens_to_ids([f'<s_{lv}_{i}>' for i in range(8192)]), dtype=np.int64)
           for lv in 'abc'}
    for lv in 'abc':
        assert (ids[lv] >= 0).all() and ids[lv].min() != tok.unk_token_id, f'level {lv} tokens missing'
    t0 = time.time()
    trie = Trie(load_catalog(a.catalog))
    meta = dict(n_sids=trie.n_sids, n_first=len(trie.first), n_prefix=len(trie.ch2),
                max_children_1=max(len(v) for v in trie.ch1.values()),
                max_children_2=max(len(v) for v in trie.ch2.values()), catalog_s=time.time() - t0)
    print(json.dumps({'stage': 'catalog', **meta}), flush=True)
    df = pd.read_parquet(a.data)
    print(json.dumps({'stage': 'data', 'rows': len(df), 'columns': list(df.columns)}), flush=True)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(df), size=min(a.users, len(df)), replace=False)
    ks = [int(x) for x in a.ks.split(',')]
    kmax = max(ks)
    beams = [int(x) for x in a.beams.split(',')]
    S = Searcher(model, {**ids}, dev, dtype)
    f = open(a.out, 'w')
    for n, ri in enumerate(rows):
        row = df.iloc[int(ri)].to_dict()
        msgs = parse_messages(row['messages'])
        prompt_msgs = [m for m in msgs if m.get('role') != 'assistant']
        targets = targets_of({'answer': [m for m in msgs if m.get('role') == 'assistant']})
        if not targets and a.target_field:
            targets = targets_of({'t': row.get(a.target_field)})
        text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text += '<|sid_begin|>'
        input_ids = tok(text, return_tensors='pt').input_ids.to(dev)
        t1 = time.time()
        S.prompt(input_ids)
        cache_lp = {}
        rec = {'row': int(ri), 'prompt_tokens': int(input_ids.shape[1]), 'targets': sorted(targets),
               'targets_in_catalog': sum(trie.contains(t) for t in targets)}
        if n < a.check:                                       # tree attention == sequential decoding for a few nodes
            a0 = int(trie.first[0]); b0 = int(trie.ch1[a0][0])
            tree = S.expand([(a0, b0)], 4)[0]
            seq = model(input_ids=torch.cat([input_ids, torch.tensor([[ids['a'][a0], ids['b'][b0]]], device=dev)], 1)).logits[0, -1:].float()
            seq = compact(seq, ids['c'])[0]
            rec['tree_check_maxabs'] = float(np.abs(tree[2] - seq[2]).max())
            rec['tree_check_lse'] = [tree[0], seq[0]]
        for norm in a.norms.split(','):
            ex = exact_search(S, trie, norm, kmax, a.chunk_a, a.chunk_p, a.max_a, a.max_p, cache_lp)
            top = ex['top']
            r = {'expanded_a': ex['exp_a'], 'expanded_p': ex['exp_p'],
                 'top32': [list(t[1]) for t in top[:32]], 'scores32': [t[0] for t in top[:32]]}
            for K in ks:
                cert = len(top) >= K and top[K - 1][0] >= ex['bound']
                r[f'cert@{K}'] = int(cert)
                if cert:
                    tau = top[K - 1][0]
                    na = sum(1 for v in ex['s1'].values() if v > tau)
                    pp = [p for p, v in ex['u2'].items() if v > tau]
                    r[f'N_A@{K}'] = na
                    r[f'N_P@{K}'] = len(pp)
                    r[f'N_Pmulti@{K}'] = sum(1 for p in pp if len(trie.ch2[p]) > 1)
            r['hit_exact@1'] = int(any(tuple(t[1]) in targets for t in top[:1]))
            r['hit_exact@32'] = int(any(tuple(t[1]) in targets for t in top[:32]))
            for B in beams:
                bm = beam_search(S, trie, norm, B, cache_lp, a.chunk_p)
                bset = [t[1] for t in bm]
                for K in (1, 10, 32):
                    if K <= B and len(top) >= K:
                        ex_set = {tuple(t[1]) for t in top[:K]}
                        r[f'beam{B}_recall@{K}'] = len(ex_set & set(bset[:K])) / K
                r[f'hit_beam{B}@1'] = int(any(t in targets for t in bset[:1]))
                r[f'hit_beam{B}@32'] = int(any(t in targets for t in bset[:32]))
            rec[norm] = r
        rec['seconds'] = time.time() - t1
        rec['nodes_scored'] = len(cache_lp)
        f.write(json.dumps(rec) + '\n')
        f.flush()
        if n % 10 == 0:
            print(json.dumps({'stage': 'progress', 'n': n + 1, 'last_s': round(rec['seconds'], 2),
                              **{f'{nm}_NA@32': rec.get(nm, {}).get('N_A@32') for nm in a.norms.split(',')}}), flush=True)
    f.close()
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
