"""Session 4 figure: catalog scaling (synthetic denser catalogs on the real model).

usage: python scripts/figures4.py RUNS_DIR OUT_DIR   (reads research/scale_catalog_office.json, RUNS/scaleA, RUNS/scaleP)
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

d = json.loads((RES / 'scale_catalog_office.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axes[0]
for mode, mk in (('siblings', 'o'), ('prefixes', 's')):
    keys = sorted((k for k in d if k.startswith(mode)), key=lambda k: int(k.split('_')[1]))
    keys = ['real_3444'] + keys
    x = [d[k]['trie']['unique_sids'] for k in keys]
    for K, col in (('10', ACC2), ('50', ACC), ('500', BASE2)):
        ax.plot(x, [d[k]['bounds'][K]['N_mean'] for k in keys], '-' + mk, color=col, ms=3, lw=1,
                label=f'minimal |N|, K={K}' if mode == 'prefixes' else None)
        ax.plot(x, [d[k]['bounds'][K]['E_mean'] for k in keys], ':' + mk, color=col, ms=2.5, lw=0.9, alpha=0.7,
                label=f'guaranteed |E|, K={K}' if mode == 'prefixes' and K == '10' else None)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('catalog size (unique SIDs)'); ax.set_ylabel('multi-child prefixes to expand (mean)')
ax.set_title('Expansion the certificate needs')
ax.text(0.98, 0.03, 'circles: siblings   squares: prefixes', transform=ax.transAxes, ha='right', fontsize=6.5, color=BASE2)
ax.legend(frameon=False, fontsize=6.5, loc='upper left')
ax = axes[1]
lat = {}
for run in ('scaleA', 'scaleP'):
    for p in sorted((RUNS / run).glob('*/results.json')):
        r = json.loads(p.read_text())
        cat = r['meta']['args'].get('catalog')
        size = 3444 if not cat else int(Path(cat).stem.split('_')[1])
        M = r['meta']['args']['n_prefix']
        lat.setdefault((run, M), []).append((size, r['latency']['device_only']['median_ms']))
for (run, M), pts in sorted(lat.items()):
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o' if run == 'scaleP' else 'o', ms=3.5,
            color=ACC if M == 16 else (ACC2 if M == 64 else BASE2), alpha=1 if run == 'scaleP' else 0.55,
            label=f'M = {M}' + (' (same allocation)' if run == 'scaleP' else ''))
ax.set_xscale('log'); ax.set_ylim(0, 8)
ax.set_xlabel('catalog size (unique SIDs)'); ax.set_ylabel('certified top-K, ms (A6000, batch 1)')
ax.set_title('Latency at a fixed expansion budget')
ax.legend(frameon=False, fontsize=6.5, loc='lower right')
fig.savefig(OUT / 'catalog_scaling.png'); fig.savefig(OUT / 'catalog_scaling.pdf'); plt.close(fig)
print('ok')
