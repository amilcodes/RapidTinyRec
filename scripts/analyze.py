"""Collect speedrun results into one table and paired-bootstrap quality deltas.

usage: python scripts/analyze.py RUNS_DIR [--baseline sweep/config] [--out results.md]
"""
import argparse, json, random, statistics
from pathlib import Path


def load(runs):
    rows = {}
    for res in sorted(Path(runs).glob('*/*/results.json')):
        key = f'{res.parent.parent.name}/{res.parent.name}'
        r = json.loads(res.read_text())
        q = res.parent / 'quality_users.json'
        rows[key] = {'res': r, 'qual_users': json.loads(q.read_text())['per_user'] if q.exists() else None}
    return rows


def boot(a, b, metric, n=2000, seed=0):
    rng = random.Random(seed)
    d = [x[metric] - y[metric] for x, y in zip(a, b)]
    m = statistics.mean(d)
    bs = []
    for _ in range(n):
        s = [d[rng.randrange(len(d))] for _ in range(len(d))]
        bs.append(statistics.mean(s))
    bs.sort()
    return m, bs[int(.025 * n)], bs[int(.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = load(a.runs)
    lines = ['| run | args | median ms | p95 ms | top1=FP32 | top10 set=FP32 | HR@5 | HR@10 | NDCG@10 | HR@50 | cert@1/5/10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for k, v in rows.items():
        r = v['res']; args = r['meta']['args']
        desc = f"{args.get('engine','beam')} {args['dtype']} {args.get('norm','full')} suf={args['suffix']} " \
               f"q={args.get('quant','none')} c={args['compile']} attn={args.get('attn','torch')}"
        if args.get('engine', 'beam') != 'beam':
            desc += f" N={args.get('n_expand')} M={args.get('n_prefix', '')}"
        lat = r.get('latency', {})
        dev = lat.get('device_only', {})
        ag = lat.get('agreement_vs_fp32_reference', {})
        qm = r.get('quality', {}).get('metrics', {})
        cert = r.get('quality', {}).get('certified_rate') or lat.get('certified_rate') or {}
        f = lambda x, p=4: (f'{x:.{p}f}' if isinstance(x, float) else '')
        lines.append(f"| {k} | {desc} | {f(dev.get('median_ms'),2)} | {f(dev.get('p95_ms'),2)} | "
                     f"{ag.get('top1_equal','')} | {ag.get('top10_set_equal','')} | {f(qm.get('HR@5'))} | {f(qm.get('HR@10'))} | "
                     f"{f(qm.get('NDCG@10'))} | {f(qm.get('HR@50'))} | "
                     f"{'/'.join(f(cert.get(t),2) for t in ('top1','top5','top10') if t in cert)} |")
    out = '\n'.join(lines)
    if a.baseline and a.baseline in rows and rows[a.baseline]['qual_users']:
        base = rows[a.baseline]['qual_users']
        out += f'\n\nPaired bootstrap (2,000 resamples over 4,866 test users) vs `{a.baseline}`:\n\n'
        out += '| run | ΔHR@10 [95% CI] | ΔNDCG@10 [95% CI] | ΔHR@5 [95% CI] |\n|---|---|---|---|\n'
        for k, v in rows.items():
            if k == a.baseline or not v['qual_users'] or len(v['qual_users']) != len(base):
                continue
            cells = []
            for m in ('HR@10', 'NDCG@10', 'HR@5'):
                mu, lo, hi = boot(v['qual_users'], base, m)
                cells.append(f'{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]')
            out += f'| {k} | ' + ' | '.join(cells) + ' |\n'
    print(out)
    if a.out:
        Path(a.out).write_text(out + '\n')


if __name__ == '__main__':
    main()
