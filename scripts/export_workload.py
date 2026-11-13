"""Export the decoding workload (prompts, targets, SID layout, catalog) for an external engine.

usage: python scripts/export_workload.py OUT_DIR       (env: REL = release dir, SPEEDRUN_DATASET)

Writes OUT_DIR/workload.json with the exact prompt token ids the released evaluation builds, the target SID
token triple per test row, the frozen 500-user latency cohort, the shared prompt prefix (what our template cache
holds), and the SID layout as level-grouped token ids; and OUT_DIR/catalog_tokens.json, the valid SIDs as token-id
triples (FlashRec builds its trie from this directly, with no assumption about token-id contiguity).
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.beam import SidCatalog

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rel = os.environ['REL']
tok = data.load_tokenizer(rel)
rows = data.test_rows(rel)
ids = data.prompts(rel, tok, rows)
targets = [list(data.sid_tokens(tok, r['item_sid'])) for r in rows]          # as in scripts/bench.py
cat = SidCatalog(tok, Path(rel) / data.CATALOG, 'cpu')
lv = cat.level_mask.long().argmax(0).tolist()
sid_ids = cat.sid_ids.tolist()
levels = [[t for t, l in zip(sid_ids, lv) if l == k] for k in range(cat.level_mask.shape[0])]
lcp = list(ids[0])
for p in ids:
    k = 0
    while k < min(len(lcp), len(p)) and lcp[k] == p[k]:
        k += 1
    lcp = lcp[:k]
uniq = sorted(set(cat.item_sid_tokens))
contig = [l == list(range(l[0], l[0] + len(l))) for l in levels]
w = {'dataset': data.DATASET, 'n_rows': len(rows), 'cohort500': data.cohort500(rows),
     'prompt_ids': [list(map(int, p)) for p in ids], 'targets': targets, 'template_ids': lcp,
     'sid_levels': levels, 'codebook_sizes': [len(l) for l in levels], 'levels_contiguous': contig,
     'unique_sids': len(uniq), 'eos_id': tok.eos_token_id}
json.dump(w, open(out / 'workload.json', 'w'))
json.dump([list(s) for s in uniq], open(out / 'catalog_tokens.json', 'w'))
print(json.dumps({k: w[k] for k in ('dataset', 'n_rows', 'codebook_sizes', 'levels_contiguous', 'unique_sids')}
                 | {'template_tokens': len(lcp), 'prompt_len': [min(map(len, ids)), max(map(len, ids))],
                    'level_ranges': [[l[0], l[-1]] for l in levels]}))
