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
    lo, hi = 1.0, max(r[1] for r in rows) * 1.15
    lx, rx, bh, gap, top = 250, width - 70, 22, 10, 24
    H = top + len(rows) * (bh + gap) + 30
    X = lambda v: lx + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (rx - lx)
    out = [f'<svg viewBox="0 0 {width} {H}" role="img" aria-label="latency ladder" class="chart">']
    for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if lo <= t <= hi:
            x = X(t)
            out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{H - 26}" class="grid"/>')
            out.append(f'<text x="{x:.1f}" y="{H - 10}" class="tick" text-anchor="middle">{t}</text>')
    for i, (lab, v, hl) in enumerate(rows):
        y = top + i * (bh + gap)
        out.append(f'<text x="{lx - 10}" y="{y + bh * 0.7:.1f}" class="lab" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{lx}" y="{y}" width="{max(1, X(v) - lx):.1f}" height="{bh}" class="{"bar hl" if hl else "bar"}"/>')
        out.append(f'<text x="{X(v) + 6:.1f}" y="{y + bh * 0.7:.1f}" class="val">{v:.1f} {unit}</text>')
    out.append(f'<text x="{(lx + rx) / 2:.0f}" y="{H}" class="tick" text-anchor="middle">median latency per request, ms (log scale)</text>')
    out.append('</svg>')
    return '\n'.join(out)


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
    for s in cfg.get('sim_tables', []):
        p = Path(s['path'])
        if p.exists():
            sims += f"<h3>{esc(s['title'])}</h3><div class=scroll>{md_table_to_html(p.read_text())}</div>"
    page = TEMPLATE.replace('{{LADDER}}', bar_chart(ladder)).replace('{{ROWS}}', '\n'.join(rows_html))
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
ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
.note { font-size: 13px; color: var(--muted); }
code { font: 13px var(--mono); background: var(--accent-soft); padding: 1px 4px; border-radius: 3px; }
</style>
<div class="wrap">
<header>
  <div class="eyebrow">MiniOneRec · Qwen2-1.5B · beam 50 · RTX A6000 · updated {{UPDATED}}</div>
  <h1>RapidTinyRec Speedrun</h1>
  <p class="lede">How fast can an LLM generative recommender decode, and is beam search even the right algorithm? Batch-1 latency on the frozen 500-user Office cohort; quality on all 4,866 Office test users.</p>
</header>
<section><h2>Findings so far</h2><ul>{{FINDINGS}}</ul></section>
<section class="panel"><h2>Latency ladder</h2><div class="scroll">{{LADDER}}</div>
<p class="note">Orange rows are this overnight run's engine; grey rows are earlier RapidTinyRec results on the same GPU and cohort. Setup, compilation and graph capture are excluded.</p></section>
<section class="panel"><h2>Speed and quality by configuration</h2><div class="scroll"><table>
<thead><tr><th>configuration</th><th class=num>median ms</th><th class=num>p95 ms</th><th class=num>HR@1</th><th class=num>HR@5</th><th class=num>HR@10</th><th class=num>NDCG@10</th><th class=num>top-10 certified</th></tr></thead>
<tbody>{{ROWS}}</tbody></table></div>
<p class="note">Quality is SID-level HR/NDCG on the full Office test set (4,866 users). "Top-10 certified" is the share of users for whom the exact decoder proves its top-10 is the global optimum of the scoring rule.</p></section>
<section class="panel"><h2>Search × scoring rule (offline, exhaustive FP32)</h2>{{SIMS}}</section>
{{NOTES}}
<section><h2>Next</h2><ul>{{NEXT}}</ul></section>
</div>
'''

if __name__ == '__main__':
    main()
