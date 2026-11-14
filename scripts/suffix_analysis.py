"""Why do the forced \\n/EOS terms hurt ranking?  usage: suffix_analysis.py DUMP_DIR RELEASE

For every user: suffix = lp(\\n|abc) + lp(EOS|abc\\n). Compare the target item's suffix with
the competitors ranked above it by lp(a)+lp(b|a)+lp(c|ab); relate the suffix to the number of
siblings under the same (a,b) prefix and to item popularity in the test histories.
"""
import collections, csv, json, re, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as S

D, meta = S.load(sys.argv[1])
rel = Path(sys.argv[2])
sid_list = [tuple(x) for x in meta['sid_tok'].tolist()]
pos_of = {s: i for i, s in enumerate(sid_list)}
tgt = torch.tensor([pos_of.get(tuple(t), -1) for t in D['targets']])
U = len(tgt)
fa, fb, fc, fs = S.terms(D, meta, 'full')
u2, _ = S.item_scores(meta, fa, fb, fc * 0, fs * 0)
u3 = u2 + fc
suf = fs                                                    # [U,S]
# siblings per SID (children of its (a,b) prefix)
pair = meta['sid_pair']
sib = torch.bincount(pair)[pair].float()                    # [S]
# popularity: occurrences in all test-history SID strings (proxy for training frequency)
tok_of = {}
rows = list(csv.DictReader(open(rel / 'Amazon/test/Office_Products_5_2016-10-2018-11.csv')))
cnt = collections.Counter()
vocab = json.loads((rel / 'Office_ckpt/tokenizer.json').read_text())['added_tokens']
tid = {t['content']: t['id'] for t in vocab}
for r in rows:
    for s in eval(r['history_item_sid']):
        cnt[tuple(tid[x] for x in re.findall(r'<[a-z]_\d+>', s))] += 1
pop = torch.tensor([cnt.get(s, 0) for s in sid_list], dtype=torch.float)
out = {}
ok = tgt >= 0
# 1. target vs competitors: among the top-10 by u3, how does the target's suffix compare?
top = u3.topk(10, -1).indices
tsuf = suf[torch.arange(U), tgt.clamp(min=0)]
csuf = suf.gather(1, top).mean(1)
in_top = (top == tgt[:, None]).any(1) & ok
out['users_target_in_top10_by_u3'] = int(in_top.sum())
out['mean_suffix_target'] = tsuf[in_top].mean().item()
out['mean_suffix_top10_competitors'] = csuf[in_top].mean().item()
out['frac_target_suffix_below_competitor_mean'] = (tsuf[in_top] < csuf[in_top]).float().mean().item()
# 2. rank change of the target when the suffix is added (exhaustive ranking)
r3 = (u3 > u3[torch.arange(U), tgt.clamp(min=0)][:, None]).sum(1)
r5 = ((u3 + suf) > (u3 + suf)[torch.arange(U), tgt.clamp(min=0)][:, None]).sum(1)
m = ok & (r3 < 50)
out['target_rank_improves_with_suffix'] = (r5[m] < r3[m]).float().mean().item()
out['target_rank_worsens_with_suffix'] = (r5[m] > r3[m]).float().mean().item()
# 3. correlates of the suffix term (pooled over users' top-50 items by u3)
t50 = u3.topk(50, -1).indices
s50 = suf.gather(1, t50).reshape(-1)
sib50 = sib[t50].reshape(-1)
pop50 = pop[t50].reshape(-1)
def corr(x, y):
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return (x * y).mean().item()
out['corr_suffix_vs_siblings'] = corr(s50, sib50)
out['corr_suffix_vs_log_popularity'] = corr(s50, torch.log1p(pop50))
out['corr_c_term_vs_siblings'] = corr(fc.gather(1, t50).reshape(-1), sib50)
out['mean_suffix_by_siblings'] = {int(k): suf.gather(1, t50).reshape(-1)[sib50 == k].mean().item()
                                  for k in sorted(set(sib50.tolist()))[:6]}
print(json.dumps(out, indent=1))
