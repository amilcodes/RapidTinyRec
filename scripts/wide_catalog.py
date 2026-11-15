"""Synthetic catalogs with a wider first level, for latency only (bench.py --catalog ... --norm trie).

The released model has 88 first-code tokens, so a wider first level cannot be scored meaningfully. Its cost can:
forward 1 carries one tree node per first code, so certified decoding's first forward grows with |A| while beam
search's later steps do not. We reuse the model's 512 second- and third-level code tokens as extra first codes and
give each new first code the subtree (second and third codes) of a randomly chosen real first code, so branching
below the first level matches the real catalog. Use trie normalization: codebook normalization of the first level
would exclude the borrowed tokens.

usage: python wide_catalog.py --catalog REAL_CATALOG --out DIR [--widths 88,176,352,600]
"""
import argparse
import collections
import random
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--widths', default='88,176,352,600')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    real = []
    for line in open(a.catalog).read().splitlines():
        if line.strip():
            parts = re.findall(r'<[a-z]_\d+>', line.split('\t')[0])[:3]
            real.append(tuple(parts))
    sub = collections.defaultdict(set)
    for a_, b_, c_ in set(real):
        sub[a_].add((b_, c_))
    firsts = sorted(sub)
    extra = [f'<b_{i}>' for i in range(256)] + [f'<c_{i}>' for i in range(256)]
    rng = random.Random(a.seed)
    rng.shuffle(extra)
    for w in (int(x) for x in a.widths.split(',')):
        assert len(firsts) <= w <= len(firsts) + len(extra), w
        sids = set(real)
        for tok in extra[:w - len(firsts)]:
            donor = rng.choice(firsts)
            sids |= {(tok, b_, c_) for b_, c_ in sub[donor]}
        sids = sorted(sids)
        path = out / f'wide_A{w}.txt'
        with open(path, 'w') as f:
            for i, (a_, b_, c_) in enumerate(sids):
                f.write(f'{a_}{b_}{c_}\tsynthetic item {i}\t{i}\n')
        print(f'{path}: {w} first codes, {len(sids)} SIDs')


if __name__ == '__main__':
    main()
