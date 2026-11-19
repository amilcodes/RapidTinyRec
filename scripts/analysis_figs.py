"""Figures from the exhaustive dumps. usage: analysis_figs.py DUMP_DIR SIM_JSON OUT_DIR [label]"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as S

dump, sim, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else 'Office'
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight'})
R = json.loads(sim.read_text())
colors = {'full': '#8a95a6', 'nosuffix': '#5b6b85', 'codebook': '#c2410c', 'trie': '#1f7a4d'}
names = {'full': 'full vocabulary + forced suffix (MiniOneRec)', 'nosuffix': 'full vocabulary, no suffix',
         'codebook': 'codebook-renormalized', 'trie': 'trie-renormalized'}
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ws = [1, 5, 10, 20, 50, 100]
for rule in ('full', 'nosuffix', 'codebook', 'trie'):
    ys = [R[f'{rule}/beam{w}']['NDCG@10'] for w in ws]
    ax.plot(ws, ys, marker='o', ms=3, color=colors[rule], label=names[rule])
    ax.axhline(R[f'{rule}/exhaustive']['NDCG@10'], color=colors[rule], ls=':', lw=1)
ax.set_xscale('log'); ax.set_xticks(ws); ax.set_xticklabels([str(w) for w in ws])
ax.set_xlabel('beam width (dotted: exact search)'); ax.set_ylabel(f'NDCG@10 ({label}, {R["users"]:,} users)')
ax.legend(fontsize=7, frameon=False)
fig.savefig(out / f'search_scoring_{label.lower()}.png'); fig.savefig(out / f'search_scoring_{label.lower()}.pdf')

D, meta = S.load(dump)
pair = meta['sid_pair']
sib = torch.bincount(pair)[pair]
ks = [k for k in range(1, 9) if (sib == k).sum() >= 10]
med = [D['lc'][:, sib == k].median().item() for k in ks]
leak = [D['cbc'][:, sib == k].median().item() for k in ks]
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.plot(ks, med, marker='o', color='#c2410c', label='median log p(c | a,b)')
ax.plot(ks, [-torch.log(torch.tensor(float(k))).item() + med[0] for k in ks], ls='--', color='#8a95a6', label='−log k (shifted)')
ax.plot(ks, leak, marker='s', color='#1f7a4d', label='median log mass on c-codebook')
ax.set_xlabel('items sharing the (a, b) prefix'); ax.set_ylabel('log-probability'); ax.legend(fontsize=7, frameon=False)
fig.savefig(out / f'sibling_penalty_{label.lower()}.png'); fig.savefig(out / f'sibling_penalty_{label.lower()}.pdf')
print('written', out)
