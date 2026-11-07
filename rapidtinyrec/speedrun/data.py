"""Prompt construction (MiniOneRec's released EvalSidDataset) and quality metrics."""
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

DATASETS = {
    'office': ('Office_ckpt', 'Office_Products'),
    'industrial': ('Industrial_ckpt', 'Industrial_and_Scientific'),
}
DATASET = os.environ.get('SPEEDRUN_DATASET', 'office')
CKPT, _NAME = DATASETS[DATASET]
TEST_CSV = f'Amazon/test/{_NAME}_5_2016-10-2018-11.csv'
CATALOG = f'Amazon/info/{_NAME}_5_2016-10-2018-11.txt'


def load_tokenizer(release):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Path(release) / CKPT, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    return tok


def test_rows(release):
    return list(csv.DictReader(open(Path(release) / TEST_CSV)))


def cohort500(rows, n=500):
    sel, seen = [], set()
    for i, r in enumerate(rows):
        if r['user_id'] not in seen:
            sel.append(i); seen.add(r['user_id'])
        if len(sel) == n:
            break
    return sel


def prompts(release, tok, rows):
    """Token ids for each row exactly as the released evaluation builds them."""
    from ..source.data import EvalSidDataset
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'rows.csv'
        with p.open('w') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        ds = EvalSidDataset(str(p), tok, max_len=2560, test=True, category='office products', K=0, seed=42)
        return [ds[i]['input_ids'] for i in range(len(rows))]


def sid_tokens(tok, sid):
    import re
    return tuple(tok.convert_tokens_to_ids(re.findall(r'<[a-z]_\d+>', sid)))


def rank_metrics(ranked, target, ks=(1, 3, 5, 10, 20, 50)):
    """HR@K / NDCG@K for one user given a ranked list of SID tuples."""
    try:
        r = ranked.index(target)
    except ValueError:
        r = None
    out = {}
    for k in ks:
        hit = r is not None and r < k
        out[f'HR@{k}'] = float(hit)
        out[f'NDCG@{k}'] = 1.0 / math.log2(r + 2) if hit else 0.0
    return out
