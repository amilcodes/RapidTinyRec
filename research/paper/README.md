# Paper (LaTeX)

`main.tex` is generated from the Markdown draft, which stays the source of truth:

```bash
python3 scripts/paper_tex.py                  # research/paper_draft.md -> research/paper/main.tex
python scripts/paper_figs.py research/paper/figures   # the two FlashRec-comparison figures (matplotlib)
cd research/paper && latexmk -pdf main.tex     # needs a TeX Live with newtx, zi4, xltabular, algorithmicx
```

`refs.bib` and the figure captions and placements in `scripts/paper_tex.py` are maintained by hand. Several
2025–26 preprints in `refs.bib` still need full titles and author lists (comments marked TODO). The figures
are copies of `research/figures/*.pdf`. To edit on Overleaf, upload this folder as a zip (New Project →
Upload Project); `main.tex` finds the figures in `figures/` or next to it.
