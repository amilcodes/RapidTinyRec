"""Does certified exact decoding scale with the catalog? Synthetic denser catalogs on the real model.

The released model's vocabulary holds 600 SID tokens (88 first codes, 256 second, 256 third), so it can score
any of 88*256*256 = 5.8M SIDs; the released catalog uses 3,444. We densify the trie and measure what the
decoder needs on real prompts (quality is meaningless on synthetic items; cost and certification are not):

  siblings   real items + new third codes under the real (a, b) prefixes  (deeper tries; up to 2,488*256)
  prefixes   real items + new (b, c) under the real first codes, b and c drawn from the real per-level code
             frequencies (more prefixes and deeper tries)

Per catalog: trie stats; certification of top-K at expansion budgets M; the guaranteed expansion set |E| (two-sided
bounds, known after forward 1) and the instance-optimal set |N| (prefixes any bound-based decoder must expand).
Catalog files are written to OUT/catalogs/ for latency runs with bench.py --catalog.

usage: scale_catalog.py --release REL --out DIR [--late-entry .. --late-entry-a ..]
"""
import argparse, collections, json, random, re, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactTrieEngine
from rapidtinyrec.speedrun import kernels
from inflight import template_of, load_bridge, batches, to_dev, forward1


def synth(real, vocab_codes, mode, size, seed=0):
    """real: list of (a, b, c) code ints; returns sorted list of unique (a, b, c), real ones included."""
    rng = random.Random(seed)
    have = set(real)
    out = set(real)
    pref = sorted({(a, b) for a, b, _ in real})
    fa = collections.Counter(a for a, _, _ in real)
    fb = collections.Counter(b for _, b, _ in real)
    fc = collections.Counter(c for _, _, c in real)
    A = sorted(fa)
    Bc, Cc = vocab_codes['b'], vocab_codes['c']
    cum = lambda w: [sum(w[:i + 1]) for i in range(len(w))]
    wb = cum([fb.get(b, 0) + 1 for b in Bc])       # add-one smoothing: unused codes still possible
    wc = cum([fc.get(c, 0) + 1 for c in Cc])
    wa = cum([fa[x] for x in A])
    cap = len(pref) * len(Cc) if mode == 'siblings' else len(A) * len(Bc) * len(Cc)
    assert size <= cap, (mode, size, cap)
    while len(out) < size:
        if mode == 'siblings':
            a, b = pref[rng.randrange(len(pref))]
            c = rng.choices(Cc, cum_weights=wc)[0]
        else:
            a = rng.choices(A, cum_weights=wa)[0]
            b = rng.choices(Bc, cum_weights=wb)[0]
            c = rng.choices(Cc, cum_weights=wc)[0]
        out.add((a, b, c))
    return sorted(out), len(have)


def write_catalog(path, sids):
    with open(path, 'w') as f:
        for i, (a, b, c) in enumerate(sids):
            f.write(f'<a_{a}><b_{b}><c_{c}>\tsynthetic item {i}\t{i}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--late-entry', default=None); ap.add_argument('--late-entry-a', default=None)
    ap.add_argument('--weights', default='1,0.75,0'); ap.add_argument('--norm', default='cbtrie')
    ap.add_argument('--users', type=int, default=500)
    ap.add_argument('--sizes', default='10000,30000,100000,300000,1000000')
    ap.add_argument('--modes', default='siblings,prefixes')
    ap.add_argument('--Ms', default='16,32,64,128,256')
    ap.add_argument('--m-max', type=int, default=512)
    a = ap.parse_args()
    out = Path(a.out); (out / 'catalogs').mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release); ids = data.prompts(a.release, tok, rows)
    real_lines = [l.split('\t')[0] for l in open(Path(a.release) / data.CATALOG).read().splitlines() if l.strip()]
    real = sorted({tuple(int(x) for x in re.findall(r'<[a-z]_(\d+)>', s)) for s in real_lines})
    vocab_codes = collections.defaultdict(list)
    for t in tok.get_vocab():
        m = re.fullmatch(r'<([a-z])_(\d+)>', t)
        if m:
            vocab_codes[m.group(1)].append(int(m.group(2)))
    vocab_codes = {k: sorted(v) for k, v in vocab_codes.items()}
    print(json.dumps({'stage': 'vocab', **{k: len(v) for k, v in vocab_codes.items()}, 'real_sids': len(real)}), flush=True)
    W = Qwen2Weights(Path(a.release) / data.CKPT, torch.float16, dev)
    assert kernels.selfcheck(torch.float16)['pass'] and kernels.selfcheck_prompt(torch.float16)['pass']
    wb, wc, ws = (float(x) for x in a.weights.split(','))
    tmpl_ids = template_of(ids)
    br_c = load_bridge(a.late_entry, W.D) if a.late_entry else None
    br_a = load_bridge(a.late_entry_a, W.D) if a.late_entry_a else None
    idx = list(range(min(a.users, len(ids))))
    Ms = [int(x) for x in a.Ms.split(',')]
    Ks = [10, 50, 100, 500]
    shared = {}

    def engine(cat, M, N):
        e = ExactTrieEngine(W, cat, n_prefix=M, n_out=N, norm=a.norm, ks=tuple(k for k in Ks if k <= N))
        e.set_weights(wb, wc, ws)
        if 'fwd' in shared:                                   # share the forward, template K/V and bridges
            e.fwd, e.tmpl, e.template_ids = shared['fwd'], shared['tmpl'], shared['tids']
            e.late, e.bridge, e.node_late_cfg = shared['late'], shared['bridge'], shared['nl']
            return e
        e.set_template(tmpl_ids)
        if br_c:
            e.set_late_entry(*br_c)
        if br_a:
            e.set_node_late(*br_a)
        e.fwd.attn = 'triton'; e.fwd.tree_fast = True
        shared.update(fwd=e.fwd, tmpl=e.tmpl, tids=getattr(e, 'template_ids', None), late=e.late, bridge=e.bridge,
                      nl=e.node_late_cfg)
        return e

    cats = [('real', len(real), Path(a.release) / data.CATALOG)]
    for mode in a.modes.split(','):
        for size in [int(x) for x in a.sizes.split(',')]:
            cap = len({(x, y) for x, y, _ in real}) * len(vocab_codes['c']) if mode == 'siblings' else 10 ** 9
            if size > cap:
                continue
            sids, _ = synth(real, vocab_codes, mode, size)
            p = out / 'catalogs' / f'{mode}_{size}.txt'
            write_catalog(p, sids)
            cats.append((mode, size, p))
    summary = {}
    for mode, size, path in cats:
        cat = SidCatalog(tok, path, dev)
        tag = f'{mode}_{size}'
        rec = {'trie': cat.stats}
        nchild = cat.v2.sum(-1).float()
        nflat = torch.where(cat.v1.view(-1), nchild[cat.pair_table.view(-1)], torch.zeros(1, device=dev))
        logn = torch.log(nflat.clamp(min=1))
        big = engine(cat, a.m_max, 500)
        multi, single = big.multi_flat, big.single_flat
        E = {K: [] for K in Ks}; Nn = {K: [] for K in Ks}; cbig = {K: 0 for K in Ks}
        with torch.no_grad():
            for chunk in batches(idx, ids, 8):
                t, ln = to_dev(chunk, ids, tok.eos_token_id, dev)
                u2 = forward1(big, t, ln, [])[0]
                r = big.run(t, ln, big.alloc(t.shape[0], t.shape[1]))
                lower = torch.cat([u2.masked_fill(~single[None], float('-inf')),
                                   (u2 - wc * logn[None]).masked_fill(~multi[None], float('-inf'))], 1)
                srt = lower.sort(1, descending=True).values
                u2m = u2.masked_fill(~multi[None], float('-inf'))
                for j, K in enumerate(Ks):
                    tauL, tau = srt[:, K - 1], r['scores'][:, K - 1]
                    ok = r['certified'][:, j]
                    E[K] += (u2m >= tauL[:, None]).sum(1).tolist()
                    Nn[K] += (u2m >= tau[:, None]).sum(1).tolist()
                    cbig[K] += int(ok.sum())
        n = len(idx)
        q = lambda v, p_: sorted(v)[min(len(v) - 1, int(p_ * len(v)))]
        rec['bounds'] = {K: {'E_mean': sum(E[K]) / n, 'E_p99': q(E[K], .99), 'N_mean': sum(Nn[K]) / n, 'N_p99': q(Nn[K], .99),
                             f'cert_M{a.m_max}': cbig[K] / n} for K in Ks}
        rec['cert'] = {}
        n_single = int((nflat == 1).sum())
        rec['single_child_prefixes'] = n_single
        for M in Ms:
            e = engine(cat, M, min(500, n_single + M * cat.W2))
            c = {K: 0 for K in Ks if K <= e.N}
            with torch.no_grad():
                for chunk in batches(idx, ids, 8):
                    t, ln = to_dev(chunk, ids, tok.eos_token_id, dev)
                    r = e.run(t, ln, e.alloc(t.shape[0], t.shape[1]))
                    for j, K in enumerate(k for k in Ks if k <= e.N):
                        c[K] += int(r['certified'][:, j].sum())
            rec['cert'][M] = {K: v / n for K, v in c.items()}
        summary[tag] = rec
        print(json.dumps({'stage': 'eval', 'catalog': tag, **rec}), flush=True)
        del big
        torch.cuda.empty_cache()
    (out / 'scale_catalog.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
