"""Paired bootstrap between two bench.py quality runs (quality_users.json, same users in the same order).

usage: python scripts/paired.py A/quality_users.json B/quality_users.json [metric ...]   (prints B - A)
"""
import json, random, sys

a = json.load(open(sys.argv[1]))['per_user']
b = json.load(open(sys.argv[2]))['per_user']
assert len(a) == len(b)
metrics = sys.argv[3:] or ['HR@1', 'HR@10', 'NDCG@10', 'HR@50']
rng = random.Random(0)
n = len(a)
idx = [[rng.randrange(n) for _ in range(n)] for _ in range(2000)]
for m in metrics:
    d = [y[m] - x[m] for x, y in zip(a, b)]
    mean = sum(d) / n
    bs = sorted(sum(d[i] for i in s) / n for s in idx)
    print(f'{m:8s} A {sum(x[m] for x in a) / n:.4f}  B {sum(y[m] for y in b) / n:.4f}  '
          f'B-A {100 * mean:+.2f} pt [{100 * bs[50]:+.2f}, {100 * bs[1949]:+.2f}]')
