"""Figures for the same-GPU comparison with FlashRec (paper section 8.1).

usage: python paper_figs.py OUT_DIR
Numbers are the medians reported in research/paper_draft.md section 8.1
(runs/frLat for latency, runs/frTp for throughput; one H100 allocation per table).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OURS, BASE, ALT = '#c1440e', '#8a96a8', '#1f4fd8'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'savefig.dpi': 200})
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

# --- latency vs quality, H100, batch 1 (runs/frLat) ----------------------------------
pts = [  # label, ms, HR@10, color, marker, label offset (points), ha
    ('FlashRec FP8 (its default)', 14.22, 0.1554, BASE, 's', (-8, 0), 'right'),
    ('FlashRec BF16 / FP16', 11.48, 0.1644, BASE, 's', (0, 8), 'center'),
    ('ours, beam-50,\nMiniOneRec semantics', 8.36, 0.1587, ALT, 'o', (0, 8), 'center'),
    ('ours, beam-50,\nFlashRec objective', 5.15, 0.1679, ALT, 'o', (8, -4), 'left'),
    ('ours, certified', 2.57, 0.1689, OURS, 'D', (8, 3), 'left'),
    ('ours, certified + SASRec term', 2.83, 0.1825, OURS, 'D', (8, -3), 'left'),
]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for label, ms, hr, col, mk, off, ha in pts:
    ax.scatter(ms, 100 * hr, color=col, marker=mk, s=34, zorder=3)
    va = 'bottom' if off[1] > 3 else ('top' if off[1] < 0 else 'center')
    ax.annotate(label, (ms, 100 * hr), xytext=off, textcoords='offset points', fontsize=7.5,
                ha=ha, va=va)
ax.scatter([11.71], [16.44], color=BASE, marker='s', s=34, zorder=3)
ax.set_xlim(0, 16)
ax.set_ylim(15.2, 18.6)
ax.set_xlabel('median latency per request (ms), H100, batch 1')
ax.set_ylabel('HR@10 (%), all 4,866 Office users')
ax.grid(alpha=0.25, lw=0.5)
fig.savefig(out / 'h100_latency_quality.pdf')
fig.savefig(out / 'h100_latency_quality.png')

# --- throughput vs batch, H100 (runs/frTp) ----------------------------------------------
B = [16, 32, 64, 128]
series = [
    ('ours, certified, dynamic FP8', B, [2391, 2896, 3077, 3148], OURS, '-', 'D'),
    ('ours, certified, FP16', B, [2310, 2559, 2678, 2681], OURS, '--', 'D'),
    ('ours, beam-50', B[:3], [457, 478, 489], ALT, '-', 'o'),
    ('FlashRec FP16', B, [408, 431, 428, 500], BASE, '-', 's'),
    ('FlashRec FP8', B, [433, 370, 363, 489], BASE, '--', 's'),
]
fig, ax = plt.subplots(figsize=(4.4, 2.8))
for label, x, y, col, ls, mk in series:
    ax.plot(x, y, ls, color=col, marker=mk, ms=4, lw=1.4, label=label,
            mfc='white' if ls == '--' else col)
ax.set_xscale('log', base=2)
ax.set_xticks(B)
ax.set_xticklabels([str(b) for b in B])
ax.set_ylim(0, 3400)
ax.set_xlabel('users per graph replay (batch)')
ax.set_ylabel('users / s, one H100')
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=7, frameon=False, loc='center right')
fig.savefig(out / 'flashrec_throughput.pdf')
fig.savefig(out / 'flashrec_throughput.png')
print('wrote', out)
