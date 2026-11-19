"""Paper figures from run results. usage: python scripts/figures.py RUNS_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})
ACC, BASE = '#c2410c', '#8a95a6'


def res(key):
    p = RUNS / key / 'results.json'
    return json.loads(p.read_text()) if p.exists() else None


def lat(key):
    r = res(key)
    return r and r.get('latency', {}).get('device_only', {}).get('median_ms')


def first(*keys):
    for k in keys:
        if res(k):
            return k
    return keys[-1]


# 1. ladder
ladder = [('HF generate, FP32', 868.8, False), ('RapidTinyRec FP32 (prior)', 114.8, False),
          ('Speedrun FP32', lat('ladder2/M0_fp32_graph'), True),
          ('RapidTinyRec BF16 (prior)', 56.5, False),
          ('Speedrun BF16, 1 graph', lat('ladder2/M1_bf16_graph'), True),
          ('+ compile', lat('ladder2/M2_bf16_compile'), True),
          ('+ fused first level + Triton attn', lat('ladder2/M3_bf16_ff_tc'), True),
          ('+ FP16 + cached template', lat('ladder2/M4_fp16_ff_tc_tmpl'), True),
          ('certified exact, trie scoring', lat('ladder2/M5_ex3_trie_tmpl'), True),
          ('certified exact, tuned codebook', lat('ladder2/M6_ex3_cb_w05_tmpl'), True)]
ladder = [l for l in ladder if l[1]]
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ys = range(len(ladder))[::-1]
ax.barh(list(ys), [l[1] for l in ladder], color=[ACC if l[2] else BASE for l in ladder])
ax.set_yticks(list(ys)); ax.set_yticklabels([l[0] for l in ladder])
ax.set_xscale('log'); ax.set_xlabel('median latency per request (ms, log scale), RTX A6000, batch 1')
for y, l in zip(ys, ladder):
    ax.text(l[1] * 1.05, y, f'{l[1]:.1f}', va='center', fontsize=8)
fig.savefig(OUT / 'ladder.png'); fig.savefig(OUT / 'ladder.pdf')

# 2. quality vs latency
pts = [('FP32 beam (ref)', 'sweepA/fp32'), ('BF16 beam', 'sweepF2/beam_ff_tc'), ('beam, no suffix', 'sweepA/bf16_nosuffix'),
       ('exact N=16 (full)', 'sweepB/ex_bf16_n16'), ('exact N=32 (full)', 'sweepB/ex_bf16_n32'),
       ('exact N=128 (full)', 'sweepB/ex_bf16_n128'), ('cert. exact, no suffix', 'sweepF2/ex3_nosuf_ff_tc'),
       ('cert. exact, trie', 'sweepF2/ex3_trie_ff_tc'), ('beam, trie', 'ladderZ/L9_beam_trie_ff_tc'),
       ('exact trie M=32', 'ladderZ/L7_ex3_trie_m32')]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for lab, k in pts:
    r = res(k)
    if not r or 'quality' not in r or 'latency' not in r:
        continue
    x, y = r['latency']['device_only']['median_ms'], r['quality']['metrics']['NDCG@10']
    ax.scatter(x, y, color=ACC if 'exact' in lab else BASE, s=22, zorder=3)
    off = {'beam, no suffix': (-6, 6), 'exact N=16 (full)': (5, -10)}.get(lab, (4, 3))
    ax.annotate(lab, (x, y), textcoords='offset points', xytext=off, fontsize=7,
                ha='right' if off[0] < 0 else 'left')
ax.set_xscale('log'); ax.set_xlabel('median latency (ms, log)'); ax.set_ylabel('NDCG@10 (4,866 Office users)')
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([15, 20, 30, 50, 100])); ax.xaxis.set_major_formatter(FixedFormatter(['15', '20', '30', '50', '100']))
ax.xaxis.set_minor_locator(NullLocator())
fig.savefig(OUT / 'frontier.png'); fig.savefig(OUT / 'frontier.pdf')

# 3. roofline
fig, ax = plt.subplots(figsize=(4.8, 3.0))
any_r = False
for q, lab in (('none', 'BF16'), ('gl_w8', 'INT8 (GemLite)'), ('gl_w4', 'INT4 (GemLite)'), ('fp8', 'FP8 (L40S)')):
    for run in ('sweepH', 'l40sG'):
        p = RUNS / run / f'roofline_{q}' / 'roofline.json'
        if p.exists():
            d = json.loads(p.read_text())
            xs = [r['M'] for r in d['results']]; ys_ = [r['median_ms'] for r in d['results']]
            ax.plot(xs, ys_, marker='o', ms=3, label=f"{lab}, {d['gpu'].replace('NVIDIA ', '')}")
            any_r = True
if any_r:
    ax.set_xscale('log'); ax.set_xlabel('tokens per forward'); ax.set_ylabel('one full forward (ms)'); ax.legend(fontsize=7)
    fig.savefig(OUT / 'roofline.png'); fig.savefig(OUT / 'roofline.pdf')

# 4. throughput
fig, ax = plt.subplots(figsize=(4.8, 3.0))
any_t = False
for k, lab in (('ladder2/M7_tp_ex3_cb_w05', 'certified exact, tuned codebook'), ('tmplJ/tp_ex3_trie_tmpl', 'certified exact, trie scoring'), ('tmplJ/tp_beam_ff_tc_tmpl', 'beam-50, full MiniOneRec scoring'),
               ('sweepI/tp_ex3_nosuf_ff_tc', 'certified exact, no suffix (no template)'), ('sweepA/bf16_compile', 'beam-50, compile only (no fusion)')):
    r = res(k)
    if r and r.get('throughput'):
        tp = r['throughput']
        bs = sorted(int(b) for b in tp)
        ax.plot(bs, [tp[str(b)]['users_per_s'] for b in bs], marker='o', ms=3, label=lab)
        any_t = True
if any_t:
    ax.set_xscale('log', base=2); ax.set_xlabel('users per batch'); ax.set_ylabel('users / s (one A6000)'); ax.legend(fontsize=7)
    fig.savefig(OUT / 'throughput.png'); fig.savefig(OUT / 'throughput.pdf')
print('figures written to', OUT)
