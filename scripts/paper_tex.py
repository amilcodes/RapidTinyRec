#!/usr/bin/env python3
"""Render the Markdown paper draft as LaTeX.

usage: python3 scripts/paper_tex.py [research/paper_draft.md] [research/paper/main.tex]
then:  cd research/paper && latexmk -pdf main.tex

research/paper_draft.md stays the source of truth. This handles the Markdown subset the draft
uses (headings, paragraphs, lists, pipe tables, code spans, emphasis, one image) and adds the
pieces Markdown cannot carry: the algorithm box, the bridge equation, figures with captions,
citations (research/paper/refs.bib) and cross-references.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'research' / 'paper_draft.md'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'research' / 'paper' / 'main.tex'

# ----------------------------------------------------------------------------------------------
# hand-written pieces
# ----------------------------------------------------------------------------------------------
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage[varqu,varl,scaled=0.95]{zi4}
\usepackage{microtype}
\usepackage{graphicx}
\graphicspath{{figures/}}  % figures/ or flat next to main.tex (e.g. an Overleaf upload)
\usepackage{booktabs,tabularx,array,xcolor,longtable,xltabular}
\setlength{\LTpre}{5pt}\setlength{\LTpost}{5pt}
\usepackage{flafter}  % floats never land above their definition (longtable-safe)
\setcounter{topnumber}{3}\setcounter{bottomnumber}{2}\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.85}\renewcommand{\bottomfraction}{0.6}
\renewcommand{\textfraction}{0.12}\renewcommand{\floatpagefraction}{0.75}
\usepackage{enumitem}
\setlist{nosep,leftmargin=1.4em,topsep=3pt}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage{newunicodechar}
\definecolor{linkblue}{HTML}{1F4FD8}
\usepackage[colorlinks,linkcolor=linkblue,citecolor=linkblue,urlcolor=linkblue,unicode]{hyperref}
\usepackage{url}
\renewcommand{\arraystretch}{1.12}
\renewcommand{\bibfont}{\small\raggedright}
\setlength{\bibsep}{3pt}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcommand{\nl}{\texttt{\textbackslash n}}
\newcommand{\code}[1]{\texttt{#1}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{≤}{\ensuremath{\le}}
\newunicodechar{≥}{\ensuremath{\ge}}
\newunicodechar{→}{\ensuremath{\to}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{∀}{\ensuremath{\forall}}
\newunicodechar{⇒}{\ensuremath{\Rightarrow}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{⟨}{\ensuremath{\langle}}
\newunicodechar{⟩}{\ensuremath{\rangle}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∪}{\ensuremath{\cup}}
\newunicodechar{⊇}{\ensuremath{\supseteq}}
\newunicodechar{†}{\textsuperscript{\textdagger}}
"""

ALGORITHM = r"""\begin{algorithm}[!htb]
\caption{Certified exact top-$K$ decoding of one request. Every step has a fixed shape ($|A|$, $M$, $N$),
so the request is one CUDA-graph replay; the certificate is computed on the device and returned with the list.}
\label{alg:certified}
\begin{algorithmic}[1]
\Require prompt $x$; catalog trie with first codes $A$, prefixes $(a,b)$ and items $(a,b,c)$; budgets $M$, $N$, $K$
\State \textbf{F1} (trie-level prefill fusion): one forward over $[x;\ \text{every } a \in A \text{ as a tree node}]$
  $\Rightarrow s(a)\ \forall a$,\ \ $s(b \mid a)$ for all valid $(a,b)$
\State $U_2(a,b) \gets s(a) + s(b \mid a)$ \Comment{admissible: score $\le U_2$}
\State \textbf{F2}: one forward over the $M$ prefixes with the largest $U_2$, token $b$ each, reusing the $a$-node K/V
  $\Rightarrow s(c \mid ab)$ for all their children
\State $U_3(a,b,c) \gets U_2(a,b) + s(c \mid ab)$ \Comment{exact if there is no forced suffix}
\If{the forced suffix is scored}
  \State \textbf{F3}: one forward over the $N$ items with the largest $U_3$, tokens $[c, \text{\nl}]$
  \State $\mathrm{exact}(a,b,c) \gets U_3 + s(\text{\nl} \mid abc) + s(\mathrm{EOS} \mid abc\,\text{\nl})$
\EndIf
\State \Return items sorted by exact score and, for each $K$, the certificate
  $\mathrm{exact}_{(K)} \ge \max\bigl(U_3 \text{ of the best un-suffixed item},\ U_2 \text{ of the best unexpanded prefix}\bigr)$
\end{algorithmic}
\end{algorithm}
"""

BRIDGE_EQ = r"""\begin{equation}
x_e(a,b) = h_e(a) + R_e\,\bigl[h_e(a);\, E[b];\, E[a];\, 1\bigr]
  + \mathrm{MLP}_e\bigl(\bigl[\mathrm{rms}(h_e(a));\, E[b];\, E[a]\bigr]\bigr),
\label{eq:bridge}
\end{equation}
"""


def fig(files, caption, label, widths=None, placement='htp'):
    widths = widths or [0.98 / len(files)] * len(files)
    body = []
    for f, w in zip(files, widths):
        if len(files) == 1:
            body.append(rf'\includegraphics[width={w:.2f}\linewidth]{{{f}}}')
        else:
            body.append(rf'\begin{{minipage}}[b]{{{w:.2f}\linewidth}}\centering'
                        rf'\includegraphics[width=\linewidth]{{{f}}}\end{{minipage}}')
    return (rf'\begin{{figure}}[{placement}]' '\n\\centering\n' + '\\hfill\n'.join(body) +
            f'\n\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n')


# figures inserted after the first block of the section with this number
FIGURES = {
    '1': fig(['h100_latency_quality.pdf'], (
        'Latency and quality on one H100 (Office, batch 1, median over $2\\times500$ requests, one allocation, '
        '\\code{runs/frLat}). FlashRec is the public SID-GR engine of \\S\\ref{sec:8.1}, ported to Qwen2 and run on the '
        'same prompts and catalog; FP16 coincides with BF16. HR@10 is measured on all 4,866 test users; the two certified '
        'points take quality from earlier runs of the same configurations.'), 'h100', [0.62]),
    '5.1': fig(['certificate_kscale.pdf'], (
        'Prefixes the certificate must expand to prove the top $K$ (median; band to p99), from exhaustive FP32 scores '
        'of all test users. Under trie or codebook normalization the cost is about $K$ prefixes; under MiniOneRec\'s '
        'full-vocabulary objective it approaches the whole trie.'), 'kscale', [0.78]),
    '5.1b': fig(['two_sided_bounds.pdf'], (
        'The guaranteed expansion set $|E|$ (Prop.~3, known right after the first forward; band to p99) against the '
        'minimal set $|N|$ that any bound-based decoder must expand (Prop.~2); means over all test users.'),
        'bounds', [0.78]),
    '5.2': fig(['late_entry_frontier.pdf'], (
        'Late entry against early exit for the third-code forward, paired against the full-depth forward (Office, all '
        'test users, 95\\% bootstrap intervals). Bridged rows that enter at layer $28-k$ stay lossless down to 2--6 '
        'remaining layers; running only the first $k$ layers loses 2.5--3.7 points of HR@1.'), 'lateentry', [0.9]),
    '6': fig(['search_scoring_office.pdf', 'sibling_penalty_office.pdf'], (
        'Left: NDCG@10 against beam width for four scoring rules (Office, all test users; dotted: exact search under '
        'the same rule). Right: the sibling penalty. The target\'s median $\\log p(c \\mid a,b)$ falls with the number '
        'of catalog items that share $(a,b)$, faster than an even split ($-\\log k$), while the mass on the $c$-codebook '
        'stays flat (exhaustive FP32 scores).'), 'sibling_penalty_office', [0.54, 0.42]),
    '6.x': fig(['value_search_error.pdf', 'fusion_office.pdf'], (
        'Left: recall of the exact top-10 under a synthetic per-item value term $\\lambda v$, $v\\sim\\mathcal{N}(0,1)$. '
        'Reranking the LLM\'s top-50 degrades as $\\lambda$ grows; certified search stays exact. Right: validation '
        'HR@10 and HR@50 against the SASRec weight $\\beta$ (Office). Reranking cannot change the top-50 set, so its '
        'HR@50 is flat.'), 'fusion', [0.36, 0.62]),
    '8': fig(['large_k_latency.pdf'], (
        'Batch-1 latency against list length $K$. Certified top-$K$ stays nearly flat up to $K=500$; beam-$K$ does '
        'not, least of all under MiniOneRec\'s full-vocabulary scoring.'), 'largek', [0.72]),
    '8.1': fig(['flashrec_throughput.pdf'], (
        'Throughput on one H100 against batch size (\\code{runs/frTp}). With beam-50 both engines serve the same '
        'number of users per second; certified decoding serves 5.4--6.4$\\times$ more.'), 'flashrec-tp', [0.6]),
}

# a float defined just before a longtable can overflow the page, so some figures wait for more blocks
FIG_AFTER = {'8.1': 4}

# Markdown images get these captions (keyed by file stem)
IMAGES = {
    'catalog_scaling': (
        'Synthetic catalogs of 10k--1M SIDs scored by the real model (500 Office test users). Left: the minimal '
        'expansion $|N|$ is flat in catalog size, while the guaranteed set $|E|$ grows with branching. Right: '
        'certified top-$K$ latency at a fixed budget $M$ (A6000, batch 1).'),
}

# raw-Markdown replacements applied to each joined paragraph / cell before conversion.
# CITE[...] / FIGREF[...] are placeholders that survive escaping.
REPLACE = [
    ('(OneRec, GR4AD, GRACE, PinRec, Spotify, NVIDIA SID-GR)',
     '(OneRec CITE[ax2506.13695], GR4AD CITE[ax2602.22732], GRACE CITE[ax2608.00938], PinRec CITE[ax2504.10507], '
     'Spotify CITE[spotify2026], NVIDIA SID-GR CITE[nvidiasidgr])'),
    ('(FlashRec: 86–92%', '(FlashRec CITE[flashrec]: 86–92%'),
    ('**Headline (Office, H100, batch 1,', '**Headline (Fig. FIGREF[h100]; Office, H100, batch 1,'),
    ('MiniOneRec Office checkpoint (Qwen2-1.5B:', 'MiniOneRec CITE[ax2510.24431] Office checkpoint (Qwen2-1.5B CITE[ax2407.10671]:'),
    ('[Stahlberg & Byrne 2019; Stahlberg et al. 2022; Fagin\'s TA; stated for GR as post-hoc beam completion in Beyond the Beam, 2609.33745.]',
     'CITE[stahlberg2019,stahlberg2022,fagin2003]; stated for GR as post-hoc beam completion in *Beyond the Beam* CITE[ax2609.33745].'),
    ('(Fig. `sibling_penalty_office`)', '(Fig. FIGREF[sibling_penalty_office])'),
    ('*Collaborative fusion.* A SASRec (', '*Collaborative fusion.* A SASRec CITE[kang2018] ('),
    ('(DASO, 0.1834) and above S-GRec, V-STAR and SA-OPSD',
     '(DASO CITE[ax2608.20611], 0.1834) and above S-GRec CITE[ax2602.10606], V-STAR CITE[ax2602.10699] and '
     'SA-OPSD CITE[ax2611.01568]'),
    ('We compare with FlashRec 0.1.2 (', 'We compare with FlashRec 0.1.2 CITE[flashrec] ('),
    ('NVIDIA\'s SID-GR example could not', 'NVIDIA\'s SID-GR example CITE[nvidiasidgr] could not'),
    ('OneRec-V2\'s lazy decoder', 'OneRec-V2\'s lazy decoder CITE[ax2508.20900]'),
    ('(CALM, Schuster et al., 2022; LayerSkip, Elhoushi et al., 2024)', '(CALM CITE[schuster2022]; LayerSkip CITE[elhoushi2024])'),
    ('Mixture-of-Depths (Raposo et al., 2024)', 'Mixture-of-Depths CITE[raposo2024]'),
    ('SkipDecode (Del Corro et al., 2023)', 'SkipDecode CITE[delcorro2023]'),
    ('EAGLE (Li et al., 2024)', 'EAGLE CITE[li2024eagle]'),
    ('2611.01072 studies beam\'s first-step losses for TIGER', 'Yao and Zhu CITE[ax2611.01072] study beam\'s first-step losses for TIGER CITE[rajput2023]'),
    ('Wu et al. (SIGIR\'25)', 'Wu et al. CITE[ax2504.09935]'),
    ('(Stahlberg & Byrne, 2019; Meister et al., 2020)', 'CITE[stahlberg2019,meister2020]'),
    ('(Stahlberg, Kulikov & Kumar, ACL\'22)', 'CITE[stahlberg2022]'),
    ('Fagin\'s threshold algorithm is', 'Fagin\'s threshold algorithm CITE[fagin2003] is'),
    ('(V-STAR, UAGR, GR4AD-X)', '(V-STAR CITE[ax2602.10699], UAGR CITE[ax2611.02981], GR4AD-X)'),
    ('{(a,b) : U2(a,b) > s_K}', 'MATH[\\{(a,b) : U_2(a,b) > s_K\\}]'),
    ('U(p) + max_{i under p} v_u(i)', 'MATH[U(p) + \\max_{i \\text{ under } p} v_u(i)]'),
]

# ----------------------------------------------------------------------------------------------
# inline conversion
# ----------------------------------------------------------------------------------------------
SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
           '~': r'\ensuremath{\sim}', '^': r'\^{}'}


def code(s):
    out = []
    for ch in s:
        if ch == '\\':
            out.append(r'\textbackslash{}')
        elif ch in '&%$#{}':
            out.append('\\' + ch)
        elif ch == '_':
            out.append(r'\_\allowbreak{}')
        elif ch == '~':
            out.append(r'\textasciitilde{}')
        elif ch == '^':
            out.append(r'\^{}')
        elif ch == '/':
            out.append(r'/\allowbreak{}')
        elif ch == '-':
            out.append('-{}')
        else:
            out.append(ch)
    return r'\code{' + ''.join(out) + '}'


SUBBASE = {'MLP': r'\mathrm{MLP}', 'exact': r'\mathrm{exact}', 'certified': r'\mathrm{certified}'}
SUBIDX = {'max': r'\max'}


def inl(s):
    keep = []

    def protect(tex):
        keep.append(tex)
        return f'\x00{len(keep) - 1}\x00'

    s = re.sub(r'`([^`]+)`', lambda m: protect(code(m.group(1))), s)
    s = re.sub(r'MATH\[(.+?)\](?=[\s.,;)]|$)', lambda m: protect('$' + m.group(1) + '$'), s)
    s = re.sub(r'\s*CITE\[([^\]]+)\]', lambda m: protect('~\\cite{' + m.group(1) + '}'), s)
    s = re.sub(r'FIGREF\[([^\]]+)\]', lambda m: protect('\\ref{fig:' + m.group(1) + '}'), s)
    s = re.sub(r'\s*\((\d{4}\.\d{5})\)', lambda m: protect('~\\cite{ax' + m.group(1) + '}'), s)
    s = re.sub(r'§\s?(\d+(?:\.\d+[a-z]?|\.[a-z])?)', lambda m: protect('\\S\\ref{sec:' + m.group(1) + '}'), s)
    # backslash escapes used in the draft
    s = s.replace('\\*', protect(r'\textsuperscript{*}')).replace('\\|', '|')
    s = s.replace('\\n', protect(r'\nl{}')).replace(' \\ ', ' ' + protect(r'$\setminus$') + ' ')
    # subscripts written as w_c, τ_L, n_max, MLP_e ...
    def sub(m):
        base, idx = m.group(1), m.group(2)
        base = SUBBASE.get(base, base)
        idx = SUBIDX.get(idx, idx)
        return protect(f'${base}_{{{idx}}}$')
    s = re.sub(r'(?<![A-Za-z\\])(MLP|exact|certified|[A-Za-zτ])_(\([A-Za-z0-9]+\)|max|[A-Za-z0-9])(?![A-Za-z0-9])', sub, s)
    s = re.sub(r'(?<![A-Za-z0-9])U([23])(?![0-9])', lambda m: protect(f'$U_{m.group(1)}$'), s)
    # emphasis (bold before italics); markers survive escaping
    s = re.sub(r'\*\*(.+?)\*\*', '\x02\\1\x03', s)
    s = re.sub(r'(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])', '\x04\\1\x05', s)
    s = s.replace('*', protect(r'\textsuperscript{*}'))
    s = re.sub(r'"([^"]+)"', '\x06\\1\x07', s)
    s = ''.join(SPECIAL.get(ch, ch) for ch in s)
    s = (s.replace('\x02', r'\textbf{').replace('\x03', '}').replace('\x04', r'\emph{')
          .replace('\x05', '}').replace('\x06', '``').replace('\x07', "''"))
    while '\x00' in s:
        s = re.sub(r'\x00(\d+)\x00', lambda m: keep[int(m.group(1))], s)
    return s


def raw(text):
    text = text.replace('\n', ' ')  # paragraph / list-item lines are joined first
    for a, b in REPLACE:
        text = text.replace(a, b)
    return text


# ----------------------------------------------------------------------------------------------
# block conversion
# ----------------------------------------------------------------------------------------------
def plain(cell):
    return re.sub(r'\*\*|`|\\(?=[*|])', '', cell)


def emwidth(cell):
    """Approximate width of a cell in em of a Times-like font."""
    w = 0.0
    for ch in plain(cell):
        if ch.isdigit():
            w += 0.5
        elif ch.islower():
            w += 0.45
        elif ch.isupper():
            w += 0.68
        elif ch in ' .,:;\'':
            w += 0.25
        elif ch in '[]()/|*':
            w += 0.33
        elif ch in '−+=<>≤≥×≈':
            w += 0.56
        elif ch in '%@—':
            w += 0.9
        else:
            w += 0.5
    return w * (1.05 if '**' in cell else 1.0)


def numeric(col):
    text = ''.join(plain(c) for c in col)
    letters = sum(ch.isalpha() for ch in text)
    return letters <= 0.25 * max(len(text), 1)


LINE_PT = 468.0  # \linewidth at 1in margins on letter paper
SIZES = ((r'\small', 10.0), (r'\footnotesize', 9.0))


def table(rows):
    def cells(line):
        parts = re.split(r'(?<!\\)\|', line.strip())
        return [c.strip() for c in parts[1:-1]]
    head, align, body = cells(rows[0]), cells(rows[1]), [cells(r) for r in rows[2:]]
    n = len(head)
    body = [(r + [''] * n)[:n] for r in body]
    right = [a.endswith(':') for a in align]
    cols = [[r[i] for r in body] for i in range(n)]
    maxw = [max(emwidth(c) for c in [head[i]] + cols[i]) for i in range(n)]
    bodyw = [sorted(emwidth(c) for c in cols[i]) for i in range(n)]
    typw = [b[int(0.75 * (len(b) - 1))] if b else 1.0 for b in bodyw]
    bodymax = [b[-1] if b else 1.0 for b in bodyw]
    plan = None
    for size, pt in SIZES:  # natural widths if the table fits
        if sum(maxw) + n * 8 / pt <= LINE_PT / pt:
            plan = (size, None)
            break
    if plan is None:
        xs = [i for i in range(n) if not right[i]] or [0]
        fixed = [i for i in range(n) if i not in xs]
        for size, pt in SIZES:
            space = LINE_PT / pt - n * 8 / pt - sum(maxw[i] for i in fixed)
            num = [i for i in xs if numeric(cols[i]) and i != 0]
            text = [i for i in xs if i not in num]
            for want in (bodymax, typw):  # numeric columns keep one line if they can
                need = sum(want[i] for i in num)
                if need + 9.0 * len(text) <= space:
                    break
            else:
                continue
            w = {i: want[i] for i in num}
            rest = space - need
            tw = sum(max(typw[i], 6.0) for i in text) or 1.0
            for i in text:
                w[i] = rest * max(typw[i], 6.0) / tw
            plan = (size, w)
            break
        if plan is None:  # scale everything down proportionally
            size, pt = SIZES[-1]
            space = LINE_PT / pt - n * 8 / pt - sum(maxw[i] for i in fixed)
            tot = sum(typw[i] for i in xs) or 1.0
            plan = (size, {i: space * typw[i] / tot for i in xs})
    size, w = plan
    if w is None:
        spec = ''.join('r' if r else 'l' for r in right)
        env_open, env_close = rf'\begin{{longtable}}{{{spec}}}', r'\end{longtable}'
    else:
        tot = sum(w.values())
        spec = ''
        for i in range(n):
            if i in w:
                spec += rf'>{{\raggedright\arraybackslash\hsize={len(w) * w[i] / tot:.3f}\hsize}}X'
            else:
                spec += 'r' if right[i] else 'l'
        env_open, env_close = rf'\begin{{xltabular}}{{\linewidth}}{{{spec}}}', r'\end{xltabular}'
    out = ['{' + size, '\\setlength{\\tabcolsep}{4pt}', env_open, '\\toprule',
           ' & '.join(inl(raw(c)) for c in head) + r' \\', '\\midrule', '\\endhead']
    for r in body:
        out.append(' & '.join(inl(raw(c)) for c in r) + r' \\')
    out += ['\\bottomrule', env_close, '}']
    return '\n'.join(out) + '\n'


def lst(items, ordered):
    env = 'enumerate' if ordered else 'itemize'
    return (f'\\begin{{{env}}}\n' + '\n'.join(r'\item ' + inl(raw(it)) for it in items) +
            f'\n\\end{{{env}}}\n')


ITEM = re.compile(r'^(\d+\.|[-*]) (.*)')


def convert(md):
    lines = md.split('\n')
    title, date, abstract, body = '', '', [], []
    cur = body
    i = 0
    pending_fig, wait = None, 0

    def emit(tex):
        nonlocal pending_fig, wait
        cur.append(tex)
        if pending_fig and cur is body:
            wait -= 1
            if wait <= 0:
                cur.append(pending_fig)
                pending_fig = None

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m = re.match(r'^(#{1,3}) (.*)', ln)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            i += 1
            if level == 1:
                title = text
                continue
            if text.lower() == 'abstract':
                cur = abstract
                continue
            cur = body
            num, name = re.match(r'^([\d]+(?:\.[\dA-Za-z]+)?)\.?\s+(.*)', text).groups()
            cmd = 'section' if level == 2 else 'subsection'
            body.append(f'\\{cmd}{{{inl(raw(name))}}}\\label{{sec:{num}}}\n')
            pending_fig, wait = FIGURES.get(num), FIG_AFTER.get(num, 1)
            continue
        if ln.startswith('```'):
            j = i + 1
            while not lines[j].startswith('```'):
                j += 1
            block = lines[i + 1:j]
            i = j + 1
            if block and block[0].startswith('input :'):
                emit(ALGORITHM)
            else:
                emit('\\begin{verbatim}\n' + '\n'.join(block) + '\n\\end{verbatim}\n')
            continue
        if ln.startswith('    ') and not ITEM.match(ln.strip()) and (i == 0 or not lines[i - 1].strip()):
            j = i
            while j < len(lines) and lines[j].startswith('    '):
                j += 1
            block = [l.strip() for l in lines[i:j]]
            i = j
            if block[0].startswith('x_e(a,b)'):
                # the sentence before ends with "a small bridge," and the one after starts with "where"
                emit(BRIDGE_EQ)
            else:
                emit('\\begin{verbatim}\n' + '\n'.join(block) + '\n\\end{verbatim}\n')
            continue
        if ln.startswith('|') and i + 1 < len(lines) and re.match(r'^\|\s*:?-', lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].startswith('|'):
                j += 1
            emit(table(lines[i:j]))
            i = j
            continue
        m = re.match(r'^!\[(.*)\]\((.*)\)', ln)
        if m:
            i += 1
            stem = Path(m.group(2)).stem
            if not any(stem in f for f in FIGURES.values()):
                emit(fig([stem + '.pdf'], IMAGES.get(stem, inl(m.group(1))), stem.replace('_', '-'), [0.9]))
            continue
        if ITEM.match(ln):
            ordered = ln[0].isdigit()
            items = []
            while i < len(lines) and lines[i].strip():
                mm = ITEM.match(lines[i])
                if mm and not lines[i].startswith(' '):
                    items.append(mm.group(2))
                elif lines[i].startswith('  '):
                    items[-1] += '\n' + lines[i].strip()
                else:
                    break
                i += 1
            emit(lst(items, ordered))
            continue
        # paragraph
        j = i
        para = []
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(('|', '#', '```')) \
                and not (ITEM.match(lines[j]) and not lines[j].startswith(' ')) and not lines[j].startswith('!['):
            para.append(lines[j])
            j += 1
        i = j
        text = raw('\n'.join(para))
        if not title or (cur is body and not body and text.startswith('*Working draft')):
            date = inl(text.strip('*'))
            continue
        prefix = '\\noindent ' if cur and cur[-1] == BRIDGE_EQ else ''
        emit(prefix + inl(text.replace('\n', ' ')) + '\n')
    return title, date, abstract, body


def main():
    md = SRC.read_text(encoding='utf8')
    title, date, abstract, body = convert(md)
    tex = [PREAMBLE, f'\\title{{{inl(title)}}}',
           '\\author{Authors to be added}',
           f'\\date{{\\small {date}}}', '\\begin{document}', '\\maketitle',
           '\\begin{abstract}', '\n'.join(abstract).strip(), '\\end{abstract}', '']
    tex += body
    tex += ['\\bibliographystyle{unsrtnat}', '\\bibliography{refs}', '\\end{document}', '']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(tex), encoding='utf8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
