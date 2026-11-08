"""Figures for the second session: certificate K-scaling, late-entry frontier, large-K latency.

usage: python scripts/figures2.py RUNS_DIR OUT_DIR   (reads research/*.json and RUNS_DIR/*/<cfg>/results.json)
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


def lat(job, cfg):
    p = RUNS / job / cfg / 'results.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())['latency']['device_only']['median_ms']


def save(fig, name):
    fig.savefig(OUT / f'{name}.png'); fig.savefig(OUT / f'{name}.pdf'); plt.close(fig)


# 1. certificate expansion set vs K
ks = json.loads((RES / 'kscale.json').read_text())
mc = json.loads((RES / 'multichild.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, ds, title in ((axes[0], 'office', 'Office'), (axes[1], 'ind', 'Industrial')):
    for rule, col, lab in (('full', BASE, 'full vocabulary (MiniOneRec)'), ('codebook', ACC2, 'codebook, w_c=0.5'),
                           ('trie', ACC, 'trie')):
        d = ks[f'{ds}/{rule}']
        K = sorted(int(k) for k in d)
        med = [d[str(k)]['pref_med'] for k in K]
        p99 = [d[str(k)]['pref_p99'] for k in K]
        ax.plot(K, med, '-o', color=col, ms=3, label=lab)
        ax.fill_between(K, med, p99, color=col, alpha=0.12, lw=0)
    d = mc[ds]
    K = sorted(int(k) for k in d)
    ax.plot(K, [d[str(k)]['multi_med'] for k in K], '--', color=ACC, lw=1.2, label='trie, multi-child only')
    ax.plot([1, 1000], [1, 1000], ':', color=BASE2, lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('K (top-K certified)'); ax.set_title(title)
axes[0].set_ylabel('prefixes to expand (median; band to p99)')
axes[0].legend(frameon=False, fontsize=7, loc='upper left')
save(fig, 'certificate_kscale')

# 2. late-entry frontier (Office; Industrial points)
rows = []
for job, sub, e in (('bridgeE22', 'bridge', 22), ('bridgeF', 'bridge', 26), ('bridgeF', 'bridge', 24),
                    ('bridgeF', 'bridge', 18), ('bridgeF', 'bridge', 14)):
    p = RUNS / job / sub / f'bridge_e{e}.json'
    if p.exists():
        rows.append((e, json.loads(p.read_text())))
ind = []
for e in (22, 18):
    p = RUNS / 'bridgeF' / 'bridge_ind' / f'bridge_e{e}.json'
    if p.exists():
        ind.append((e, json.loads(p.read_text())))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, metric, ylab in ((axes[0], 'dHR@1_vs_full', 'Δ HR@1 vs full depth (pt)'), (axes[1], 'dNDCG@10_vs_full', 'Δ NDCG@10 vs full depth (pt)')):
    for variant, col, lab, mk in (('mlp', ACC, 'late entry, distilled bridge', 'o'), ('ridge', ACC2, 'late entry, ridge bridge', 's'),
                                  ('exit', BASE, 'early exit (first layers)', '^')):
        pts = sorted((28 - e, r[variant][metric]) for e, r in rows)
        x = [p[0] for p in pts]
        m = [100 * p[1][0] for p in pts]
        lo = [100 * (p[1][0] - p[1][1]) for p in pts]
        hi = [100 * (p[1][2] - p[1][0]) for p in pts]
        ax.errorbar(x, m, yerr=[lo, hi], fmt='-' + mk, color=col, ms=4, capsize=2, lw=1.2, label=lab)
    if ind:
        pts = sorted((28 - e, r['mlp'][metric]) for e, r in ind)
        ax.errorbar([p[0] for p in pts], [100 * p[1][0] for p in pts],
                    yerr=[[100 * (p[1][0] - p[1][1]) for p in pts], [100 * (p[1][2] - p[1][0]) for p in pts]],
                    fmt='D', mfc='none', color=ACC, ms=5, capsize=2, label='distilled bridge, Industrial')
    ax.axhline(0, color=BASE2, lw=0.6)
    ax.set_xlabel('layers run by the third-code forward (of 28)'); ax.set_ylabel(ylab)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=7, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(bottom=0.24, wspace=0.3)
save(fig, 'late_entry_frontier')

# 3. large-K latency
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, job, gpu in ((axes[0], 'largeK', 'RTX A6000'), (axes[1], 'largeKH', 'H100')):
    K = [10, 50, 100, 200, 500]
    cert = [lat(job, f'T{k}') for k in K]
    bt = [None, lat(job, 'Bt50'), lat(job, 'Bt100'), lat(job, 'Bt200'), lat(job, 'Bt500')]
    bf = [None, None, lat(job, 'Bf100'), None, lat(job, 'Bf500')]
    ax.plot(K, cert, '-o', color=ACC, ms=4, label='certified top-K (trie, two forwards)')
    kk = [k for k, v in zip(K, bt) if v]; ax.plot(kk, [v for v in bt if v], '-s', color=ACC2, ms=4, label='beam-K, trie scoring')
    kk = [k for k, v in zip(K, bf) if v]; ax.plot(kk, [v for v in bf if v], '-^', color=BASE, ms=4, label='beam-K, MiniOneRec scoring')
    ax.set_xscale('log'); ax.set_xlabel('K'); ax.set_ylabel('median ms, batch 1'); ax.set_title(gpu); ax.set_ylim(0, None)
axes[0].legend(frameon=False, fontsize=7, loc='upper left')
save(fig, 'large_k_latency')
print('wrote', sorted(p.name for p in OUT.glob('*.png')))
