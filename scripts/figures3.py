"""Figures for session 3: guaranteed expansion set vs K, value-augmented search error, collaborative fusion.

usage: python scripts/figures3.py RUNS_DIR OUT_DIR
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RES = Path(__file__).resolve().parents[1] / 'research'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})
ACC, ACC2, BASE, BASE2 = '#c2410c', '#1d4ed8', '#8a95a6', '#374151'


def save(fig, name):
    fig.savefig(OUT / f'{name}.png'); fig.savefig(OUT / f'{name}.pdf'); plt.close(fig)


# 1. guaranteed expansion set |E| (two-sided bounds) vs minimal |N| vs K
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, tag, title in ((axes[0], 'office_cas', 'Office (late entry, cbtrie 0.75)'), (axes[1], 'ind_cbtrie', 'Industrial (cbtrie 0.75)')):
    p = RES / f'lbound_{tag}.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())['K']
    K = sorted(int(k) for k in d)
    ax.plot(K, [d[str(k)]['E_mean'] for k in K], '-o', color=ACC, ms=3, label='guaranteed set |E| (mean)')
    ax.fill_between(K, [d[str(k)]['E_mean'] for k in K], [d[str(k)]['E_p99'] for k in K], color=ACC, alpha=0.12, lw=0)
    ax.plot(K, [d[str(k)]['N_mean'] for k in K], '-s', color=ACC2, ms=3, label='minimal set |N| (mean)')
    ax.plot([1, 500], [1, 500], ':', color=BASE2, lw=0.8, label='y = K')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('K'); ax.set_title(title)
axes[0].set_ylabel('multi-child prefixes to expand')
axes[0].legend(frameon=False, fontsize=7, loc='upper left')
save(fig, 'two_sided_bounds')

# 2. value-augmented objective: search error of "LLM top-50 + rerank" vs exact
p = RUNS / 'valS' / 'office' / 'value_search.json'
if p.exists():
    v = json.loads(p.read_text())['bids']
    lam = sorted(float(k) for k in v)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(lam, [100 * v[str(l) if str(l) in v else repr(l)]['rerank50_recall@10_of_exact'] for l in lam], '-o', color=BASE, ms=4,
            label='LLM top-50, then rerank')
    ax.axhline(100, color=ACC, lw=1.2, label='certified exact (2 forwards)')
    ax.set_xlabel('value weight λ (v ~ N(0,1) per item)'); ax.set_ylabel('recall of exact top-10 (%)')
    ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=7, loc='lower left')
    save(fig, 'value_search_error')

# 3. collaborative fusion: validation sweep
for tag, job in (('office', 'fuseC'), ('industrial', 'fuseI')):
    p = RUNS / job / f'fuse_{tag}.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())['valid']
    b = sorted(float(k) for k in d)
    key = lambda x: str(x) if str(x) in d else repr(x)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, m in ((axes[0], 'HR@10'), (axes[1], 'HR@50')):
        ax.plot(b, [100 * d[key(x)]['exact'][m] for x in b], '-o', color=ACC, ms=3, label='certified exact, LLM + β·log q_SASRec')
        ax.plot(b, [100 * d[key(x)]['rerank50'][m] for x in b], '-s', color=BASE, ms=3, label='LLM top-50, then rerank')
        ax.set_xlabel('β'); ax.set_ylabel(f'validation {m} (%)')
    axes[0].legend(frameon=False, fontsize=7, loc='lower right')
    save(fig, f'fusion_{tag}')
print('wrote', sorted(p.name for p in OUT.glob('*.png')))
