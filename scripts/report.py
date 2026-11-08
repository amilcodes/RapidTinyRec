"""Build research/report.html from run results (static page, inline SVG charts).

usage: python scripts/report.py RUNS_DIR CONFIG_JSON OUT_HTML
CONFIG_JSON lists the ladder rungs, comparison rows and optional simulation tables.
"""
import html, json, sys
from pathlib import Path


def esc(x):
    return html.escape(str(x))


def get(runs, key):
    p = Path(runs) / key / 'results.json'
    return json.loads(p.read_text()) if p.exists() else None


def bar_chart(rows, width=760, unit='ms'):
    """Horizontal bars on a log scale: rows = [(label, value, highlight)]."""
    import math
    rows = [r for r in rows if r[1]]
    if not rows:
        return ''
    lo, hi = 10.0, max(r[1] for r in rows) * 1.15
    lx, rx, bh, gap, top = 250, width - 70, 22, 10, 24
    H = top + len(rows) * (bh + gap) + 44
    X = lambda v: lx + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (rx - lx)
    out = [f'<svg viewBox="0 0 {width} {H}" role="img" aria-label="latency ladder" class="chart">']
    for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if lo <= t <= hi:
            x = X(t)
            out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{H - 38}" class="grid"/>')
            out.append(f'<text x="{x:.1f}" y="{H - 24}" class="tick" text-anchor="middle">{t}</text>')
    for i, (lab, v, hl) in enumerate(rows):
        y = top + i * (bh + gap)
        out.append(f'<text x="{lx - 10}" y="{y + bh * 0.7:.1f}" class="lab" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{lx}" y="{y}" width="{max(1, X(v) - lx):.1f}" height="{bh}" class="{"bar hl" if hl else "bar"}"/>')
        out.append(f'<text x="{X(v) + 6:.1f}" y="{y + bh * 0.7:.1f}" class="val">{v:.1f} {unit}</text>')
    out.append(f'<text x="{(lx + rx) / 2:.0f}" y="{H - 4}" class="tick" text-anchor="middle">median latency per request, ms (log scale)</text>')
    out.append('</svg>')
    return '\n'.join(out)


def line_chart(sim, width=760, height=300):
    """NDCG@10 vs beam width per scoring rule, dotted exact-search level, from simulate.py output."""
    import math
    rules = [('full', 'MiniOneRec scoring', 'c1'), ('nosuffix', 'no forced suffix', 'c2'),
             ('codebook', 'codebook softmax', 'c3'), ('trie', 'trie softmax', 'c4')]
    ws = [5, 10, 20, 50, 100]
    vals = [sim[f'{r}/beam{w}']['NDCG@10'] for r, _, _ in rules for w in ws] + [sim[f'{r}/exhaustive']['NDCG@10'] for r, _, _ in rules]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12 or 0.001
    lo, hi = lo - pad, hi + pad
    L, R, T, B = 62, width - 190, 16, height - 40
    X = lambda w: L + (math.log10(w) - math.log10(ws[0])) / (math.log10(ws[-1]) - math.log10(ws[0])) * (R - L)
    Y = lambda v: B - (v - lo) / (hi - lo) * (B - T)
    o = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="NDCG@10 versus beam width" class="chart">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        o.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}" class="grid"/>')
        o.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.3f}</text>')
    for w in ws:
        o.append(f'<text x="{X(w):.1f}" y="{B+16}" class="tick" text-anchor="middle">{w}</text>')
    o.append(f'<text x="{(L+R)/2:.0f}" y="{height-4}" class="tick" text-anchor="middle">beam width (dotted line: certified exact search)</text>')
    for i, (r, lab, cls) in enumerate(rules):
        pts = ' '.join(f'{X(w):.1f},{Y(sim[f"{r}/beam{w}"]["NDCG@10"]):.1f}' for w in ws)
        o.append(f'<polyline points="{pts}" class="ln {cls}" fill="none"/>')
        for w in ws:
            o.append(f'<circle cx="{X(w):.1f}" cy="{Y(sim[f"{r}/beam{w}"]["NDCG@10"]):.1f}" r="2.6" class="pt {cls}"/>')
        ye = Y(sim[f'{r}/exhaustive']['NDCG@10'])
        o.append(f'<line x1="{L}" y1="{ye:.1f}" x2="{R}" y2="{ye:.1f}" class="ex {cls}"/>')
        ly = T + 8 + i * 20
        o.append(f'<line x1="{R+14}" y1="{ly}" x2="{R+34}" y2="{ly}" class="ln {cls}"/>')
        o.append(f'<text x="{R+40}" y="{ly+4}" class="lab">{esc(lab)}</text>')
    o.append('</svg>')
    return '\n'.join(o)


def main():
    runs, cfg_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = json.loads(Path(cfg_path).read_text())
    ladder = []
    for r in cfg['ladder']:
        v = r.get('ms')
        if v is None and r.get('run'):
            res = get(runs, r['run'])
            v = res and res.get('latency', {}).get('device_only', {}).get('median_ms')
        ladder.append((r['label'], v, r.get('ours', False)))
    rows_html = []
    for r in cfg['compare']:
        res = get(runs, r['run'])
        if not res:
            continue
        lat = res.get('latency', {}).get('device_only', {})
        q = res.get('quality', {}).get('metrics', {})
        cert = res.get('quality', {}).get('certified_rate') or {}
        f = lambda v, p=4: f'{v:.{p}f}' if isinstance(v, (int, float)) else '—'
        rows_html.append(
            f"<tr><td>{esc(r['label'])}</td><td class=num>{f(lat.get('median_ms'), 1)}</td>"
            f"<td class=num>{f(lat.get('p95_ms'), 1)}</td><td class=num>{f(q.get('HR@1'))}</td><td class=num>{f(q.get('HR@5'))}</td>"
            f"<td class=num>{f(q.get('HR@10'))}</td><td class=num>{f(q.get('NDCG@10'))}</td>"
            f"<td class=num>{f(cert.get('top10'), 2) if cert else '—'}</td></tr>")
    sims = ''
    if cfg.get('sim_json') and Path(cfg['sim_json']).exists():
        sim = json.loads(Path(cfg['sim_json']).read_text())
        sims += f"<p class=note>{sim['users']:,} Office test users, exhaustive FP32 scores of all 3,444 SIDs per user.</p><div class=scroll>{line_chart(sim)}</div>"
    for s in cfg.get('sim_tables', []):
        p = Path(s['path'])
        if p.exists():
            sims += f"<h3>{esc(s['title'])}</h3><div class=scroll>{md_table_to_html(p.read_text())}</div>"
    ladder_h = []
    for r in cfg.get('ladder_h100', []):
        v = r.get('ms')
        if v is None and r.get('run'):
            res = get(runs, r['run'])
            v = res and res.get('latency', {}).get('device_only', {}).get('median_ms')
        ladder_h.append((r['label'], v, r.get('ours', False)))
    figs = ''
    import base64
    for f in cfg.get('figures', []):
        fp = Path(f['path'])
        if fp.exists():
            b64 = base64.b64encode(fp.read_bytes()).decode()
            figs += (f"<section class=panel><h2>{esc(f['title'])}</h2><img alt=\"{esc(f['title'])}\" "
                     f"src=\"data:image/png;base64,{b64}\" style=\"width:100%;height:auto;background:#fff;border-radius:4px\">"
                     f"<p class=note>{f.get('note', '')}</p></section>")
    page = TEMPLATE.replace('{{LADDER}}', bar_chart(ladder)).replace('{{ROWS}}', '\n'.join(rows_html))
    page = page.replace('{{LADDER_H100}}', bar_chart(ladder_h) if ladder_h else '').replace('{{FIGS}}', figs)
    page = page.replace('{{SIMS}}', sims).replace('{{FINDINGS}}', '\n'.join(f'<li>{x}</li>' for x in cfg.get('findings', [])))
    page = page.replace('{{UPDATED}}', esc(cfg.get('updated', ''))).replace('{{NEXT}}', '\n'.join(f'<li>{x}</li>' for x in cfg.get('next', [])))
    page = page.replace('{{NOTES}}', cfg.get('notes', ''))
    Path(out).write_text(page)


def md_table_to_html(md):
    lines = [l for l in md.strip().splitlines() if l.startswith('|')]
    if len(lines) < 2:
        return ''
    cells = lambda l: [c.strip() for c in l.strip().strip('|').split('|')]
    head = cells(lines[0])
    body = [cells(l) for l in lines[2:]]
    h = '<table><thead><tr>' + ''.join(f'<th>{esc(c)}</th>' for c in head) + '</tr></thead><tbody>'
    for r in body:
        h += '<tr>' + ''.join(f'<td class="{"num" if i else ""}">{esc(c)}</td>' for i, c in enumerate(r)) + '</tr>'
    return h + '</tbody></table>'


TEMPLATE = r'''<title>RapidTinyRec Speedrun</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; charts and wide tables scroll inside their own frames. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #161a21; --muted: #5a6371; --rule: #dfe3ea;
  --accent: #c2410c; --accent-soft: #fbe3d6; --base: #8a95a6; --good: #1f7a4d;
  --display: "Instrument Sans", "Helvetica Neue", Arial, sans-serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1216; --surface: #171b21; --ink: #e7eaef; --muted: #9aa3b1; --rule: #2a3039;
  --accent: #f08a54; --accent-soft: #3a2418; --base: #6c7686; --good: #5cc98d; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1216; --surface: #171b21; --ink: #e7eaef; --muted: #9aa3b1; --rule: #2a3039;
  --accent: #f08a54; --accent-soft: #3a2418; --base: #6c7686; --good: #5cc98d; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); }
.wrap { max-width: 960px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 28px; }
header { display: grid; gap: 10px; }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font: 700 clamp(28px, 5vw, 42px)/1.1 var(--display); margin: 0; text-wrap: balance; }
h2 { font: 600 22px/1.25 var(--display); margin: 0 0 8px; text-wrap: balance; }
h3 { font: 600 16px/1.3 var(--display); margin: 18px 0 6px; }
p, li { max-width: 68ch; }
.lede { font-size: 17px; color: var(--muted); margin: 0; }
section { display: grid; gap: 10px; min-width: 0; }
.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px; min-width: 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: left; white-space: nowrap; }
th { font: 500 11.5px/1.2 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
td.num, th.num { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
.chart { width: 100%; min-width: 620px; height: auto; display: block; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .tick { fill: var(--muted); font: 11px var(--mono); }
.chart .lab { fill: var(--ink); font: 12.5px var(--body); }
.chart .val { fill: var(--ink); font: 12px var(--mono); }
.chart .bar { fill: var(--base); }
.chart .bar.hl { fill: var(--accent); }
.chart .ln { stroke-width: 2; fill: none; }
.chart .ex { stroke-width: 1; stroke-dasharray: 3 3; }
.chart .c1 { stroke: var(--base); fill: var(--base); }
.chart .c2 { stroke: var(--ink); fill: var(--ink); }
.chart .c3 { stroke: var(--accent); fill: var(--accent); }
.chart .c4 { stroke: var(--good); fill: var(--good); }
.chart polyline.c1, .chart polyline.c2, .chart polyline.c3, .chart polyline.c4 { fill: none; }
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
.note { font-size: 13px; color: var(--muted); }
code { font: 13px var(--mono); background: var(--accent-soft); padding: 1px 4px; border-radius: 3px; }
</style>
<div class="wrap">
<header>
  <div class="eyebrow">MiniOneRec · Qwen2-1.5B · RTX A6000 + H100 · updated {{UPDATED}}</div>
  <h1>RapidTinyRec Speedrun</h1>
  <p class="lede">How fast can an LLM generative recommender decode, and is beam search even the right algorithm? Batch-1 latency on the frozen 500-user Office cohort; quality on all 4,866 Office and 4,533 Industrial test users.</p>
</header>
<section><h2>Findings so far</h2><ul>{{FINDINGS}}</ul></section>
<section class="panel"><h2>Latency ladder</h2><div class="scroll">{{LADDER}}</div>
<p class="note">Orange rows are the speedrun engine; grey rows are earlier RapidTinyRec results on the same GPU and cohort. Setup, compilation and graph capture are excluded.</p></section>
<section class="panel"><h2>Latency ladder, H100</h2><div class="scroll">{{LADDER_H100}}</div>
<p class="note">Same cohort and engine on one H100 (gpu-debug partition). The FP32 rung is the speedrun engine; Hugging Face was not re-measured on this GPU.</p></section>
{{FIGS}}
<section class="panel"><h2>Speed and quality by configuration</h2><div class="scroll"><table>
<thead><tr><th>configuration</th><th class=num>median ms</th><th class=num>p95 ms</th><th class=num>HR@1</th><th class=num>HR@5</th><th class=num>HR@10</th><th class=num>NDCG@10</th><th class=num>top-10 certified</th></tr></thead>
<tbody>{{ROWS}}</tbody></table></div>
<p class="note">Quality is SID-level HR/NDCG on the full test set (4,866 Office or 4,533 Industrial users). "Top-10 certified" is the share of users for whom the exact decoder proves its top-10 is the global optimum of its scoring rule. Latency comes from each configuration's own run on an RTX A6000; the no-suffix and codebook beam rows were measured with an earlier engine stage (no compile, no first-level fusion) and would run near 14 ms with the final engine.</p></section>
<section class="panel"><h2>Search × scoring rule (offline, exhaustive FP32)</h2>{{SIMS}}</section>
{{NOTES}}
<section><h2>Next</h2><ul>{{NEXT}}</ul></section>
</div>
'''

if __name__ == '__main__':
    main()
