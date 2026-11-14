"""Which prefixes does the certificate need, and can they be predicted before the first forward?

usage: speculate.py VALID_DUMP TEST_DUMP VALID_CSV TEST_CSV [--out res.json] [--wc 0.5]

Objective: codebook-normalized s(a) + s(b|a) + wc * s(c|ab)   (all terms <= 0).
Needed sets for top-K certification of the *test* request of each user:
  A_K = {a : s(a) > s_K}            first codes whose subtree may still hold a top-K item
  P_K = {(a,b) : U2(a,b) > s_K}     prefixes that must be expanded with a third-code forward
Predictors of P_K that are available before any forward of the test request:
  prev   - the same user's previous request (MiniOneRec validation sample = history minus the
           last item), prefixes ranked by that request's U2
  hist   - prefixes of the items in the prompt's history, most recent first
  pop    - prefixes ranked by mean U2 over validation users (user-independent)
  mix    - prev top half + hist + pop fill
A predictor "hits" at budget M if P_K is contained in its top-M prefixes: the request then
certifies with ONE forward (prompt + all first codes + M speculative prefix nodes).
"""
import argparse, ast, csv, json, re, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as S

csv.field_size_limit(10**9)


def rows_of(p):
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('valid'); ap.add_argument('test'); ap.add_argument('valid_csv'); ap.add_argument('test_csv')
    ap.add_argument('--out'); ap.add_argument('--wc', type=float, default=0.5); ap.add_argument('--ckpt', required=True)
    a = ap.parse_args()
    Dv, meta = S.load(a.valid)
    Dt, _ = S.load(a.test)
    vrows, trows = rows_of(a.valid_csv), rows_of(a.test_csv)
    t_index = {r['user_id']: i for i, r in enumerate(trows)}
    t_dump_pos = {int(r): i for i, r in enumerate(Dt['rows'].tolist())}
    pairs = []  # (valid dump idx, test dump idx, test csv row)
    for vi, r in enumerate(Dv['rows'].tolist()):
        uid = vrows[r]['user_id']
        tr = t_index.get(uid)
        if tr is not None and tr in t_dump_pos:
            pairs.append((vi, t_dump_pos[tr], tr))
    print(f'matched users: {len(pairs)} of {len(Dv["rows"])}', file=sys.stderr)
    vi = torch.tensor([p[0] for p in pairs]); ti = torch.tensor([p[1] for p in pairs])

    sav, sbv, scv, _ = S.terms(Dv, meta, 'codebook')
    sat, sbt, sct, _ = S.terms(Dt, meta, 'codebook')
    sav, sbv = sav[vi], sbv[vi]
    sat, sbt, sct = sat[ti], sbt[ti], sct[ti]
    U, nA, W1 = sbt.shape
    pair = meta['sid_pair']; flat = meta['sid_flat']
    P2 = int(pair.max()) + 1
    # prefix id -> flat (a, slot) index, so prefix U2 can be read from [U, nA*W1]
    p2flat = torch.zeros(P2, dtype=torch.long).scatter_(0, pair, flat)
    p2a = p2flat // W1

    def u2p(sa, sb):
        return (sa[..., None] + sb).reshape(sa.shape[0], -1)[:, p2flat]   # [U, P2]

    U2t, U2v = u2p(sat, sbt), u2p(sav, sbv)
    tot = U2t[:, pair] + a.wc * sct                                      # [U, S]
    srt = tot.sort(-1, descending=True).values

    # SID string -> prefix id through the tokenizer (catalog tokens are in meta['sid_tok'])
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
    ab_to_pid = {}
    for i, (ta, tb, _) in enumerate(meta['sid_tok'].tolist()):
        ab_to_pid[(ta, tb)] = int(pair[i])

    def pid_of(sid):
        codes = re.findall(r'<[a-z]_\d+>', sid)[:2]
        return ab_to_pid.get(tuple(tk.convert_tokens_to_ids(codes)))

    pop_rank = U2v.mean(0).argsort(descending=True)                      # user-independent
    res = {'users': U, 'wc': a.wc, 'prefixes': P2, 'first_codes': nA, 'hist_prefix_coverage': None}
    Ks = (1, 10, 20, 50)
    Ms = (8, 16, 32, 48, 64, 96, 128)
    hist_lists, miss_map = [], 0
    for _, _, tr in pairs:
        hs = ast.literal_eval(trows[tr]['history_item_sid'])[::-1]
        out = []
        for s in hs:
            pid = pid_of(s)
            if pid is None:
                miss_map += 1
            elif pid not in out:
                out.append(pid)
        hist_lists.append(out)
    res['hist_prefix_coverage'] = 1 - miss_map / max(1, sum(len(ast.literal_eval(trows[t]['history_item_sid'])) for _, _, t in pairs))

    prev_rank = U2v.argsort(-1, descending=True)                         # [U, P2]

    def ranked(name, u):
        if name == 'prev':
            return prev_rank[u].tolist()
        if name == 'pop':
            return pop_rank.tolist()
        if name == 'hist':
            h = hist_lists[u]
            return h + [p for p in pop_rank.tolist() if p not in set(h)]
        if name == 'mix':
            seen, out = set(), []
            pr = prev_rank[u].tolist()
            for src in (pr[:16], hist_lists[u], pr[16:]):
                for p in src:
                    if p not in seen:
                        seen.add(p); out.append(p)
            return out
        raise ValueError(name)

    R = {name: [ranked(name, u)[:max(Ms)] for u in range(U)] for name in ('prev', 'hist', 'pop', 'mix')}
    need_stats = {}
    for K in Ks:
        thr = srt[:, K - 1:K]
        needP = U2t > thr                                                # [U, P2]
        maxA = sat.max(-1, keepdim=True).values
        needA = sat > thr
        nP, nAk = needP.sum(1).float(), needA.sum(1).float()
        q = torch.tensor([.5, .9, .99])
        need_stats[K] = {'prefixes': dict(zip(('median', 'p90', 'p99'), torch.quantile(nP, q).tolist())) | {'mean': nP.mean().item()},
                         'first_codes': dict(zip(('median', 'p90', 'p99'), torch.quantile(nAk, q).tolist())) | {'mean': nAk.mean().item()}}
        for name in ('prev', 'hist', 'pop', 'mix'):
            for M in Ms:
                hit, resid = 0, 0
                for u in range(U):
                    top = set(R[name][u][:M])
                    need = set(torch.nonzero(needP[u]).flatten().tolist())
                    miss = need - top
                    hit += not miss
                    resid += len(miss)
                need_stats[K][f'{name}@{M}'] = {'hit': hit / U, 'mean_missing': resid / U}
    res['need'] = need_stats
    # oracle: how often the previous request's certified top-K set equals the new one
    res['prev_top10_overlap'] = ((tot.topk(10, -1).indices[:, :, None] == (U2v[:, pair] + a.wc * scv[vi]).topk(10, -1).indices[:, None, :]).any(-1).float().sum(1) / 10).mean().item()
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
