# Paper

- `main.tex` / `main.pdf`: the submission draft (ACM `sigconf`, anonymous), written by hand.
  Build: `latexmk -pdf main.tex` (TeX Live with acmart; on Oscar: `module load texlive/20240312-xnyd`).
- `lab_log.tex`: every result, generated from `research/paper_draft.md` by `scripts/paper_tex.py`
  (the long-form record; not for submission).
- `refs.bib`: shared bibliography. Several 2025–26 preprints still need full titles and author lists
  (comments marked TODO).
- `figures/`: copies of `research/figures/*.pdf`; `scripts/paper_figs.py` makes the two FlashRec figures.
