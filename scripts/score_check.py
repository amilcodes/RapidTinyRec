"""Which objective does an engine's beam score correspond to? Re-score its returned SIDs with the HF reference model
(FP32) under each normalization and compare with the engine's reported scores.

usage: python scripts/score_check.py WORKLOAD_DIR CKPT OUT.json RUN_A/latency_users.json [RUN_B/... ...]
Normalizations (per step, summed over the 3 codes):
  full      log-softmax over the whole vocabulary (MiniOneRec / HF)
  codebook  over the current level's SID tokens (FlashRec's documented per-codebook head)
  sid600    over all SID tokens of every level (FlashRec's restricted head without the per-level split)
  trie      over the valid trie children of the prefix (mask-then-renormalize)
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

wd, ck, outp, runs = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
W = json.loads((wd / 'workload.json').read_text())
cat = [tuple(x) for x in json.loads((wd / 'catalog_tokens.json').read_text())]
kids = {}
for s in cat:
    for i in range(3):
        kids.setdefault(s[:i], set()).add(s[i])
levels = [torch.tensor(l) for l in W['sid_levels']]
sid_all = torch.cat(levels)
m = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).cuda().eval()
R = [{r['row']: r for r in json.loads(Path(p).read_text())} for p in runs]
rows = [r for r in R[0] if all(r in x for x in R)][:12]


@torch.no_grad()
def scores(prompt, sid):
    ids = torch.tensor([prompt + list(sid[:2])], device='cuda')
    lg = m(ids).logits[0, -3:].float()          # predictions for code 0, 1, 2
    out = {k: 0.0 for k in ('full', 'codebook', 'sid600', 'trie')}
    out['cb_steps'] = []
    for i in range(3):
        l = lg[i]; t = sid[i]
        out['full'] += (l[t] - torch.logsumexp(l, 0)).item()
        cb = (l[t] - torch.logsumexp(l[levels[i].cuda()], 0)).item()
        out['codebook'] += cb; out['cb_steps'].append(cb)
        out['sid600'] += (l[t] - torch.logsumexp(l[sid_all.cuda()], 0)).item()
        ch = torch.tensor(sorted(kids[tuple(sid[:i])]), device='cuda')
        out['trie'] += (l[t] - torch.logsumexp(l[ch], 0)).item()
    return out


res = []
for row in rows:
    P = W['prompt_ids'][row]
    for ri, (p, Rr) in enumerate(zip(runs, R)):
        r = Rr[row]
        for k in range(3):
            sid = tuple(r['sids'][k])
            res.append({'row': row, 'run': p, 'rank': k, 'sid': sid, 'reported': r['scores'][k], **scores(P, sid)})
json.dump(res, open(outp, 'w'), indent=0)
import statistics
for p in runs:
    sub = [x for x in res if x['run'] == p]
    print(p, {k: round(statistics.mean(abs(x['reported'] - x[k]) for x in sub), 4) for k in ('full', 'codebook', 'sid600', 'trie')})
    drop = {f'cb_without_step{j}': round(statistics.mean(abs(x['reported'] - (x['codebook'] - x['cb_steps'][j])) for x in sub), 4)
            for j in range(3)}
    print(p, drop)
