#!/bin/bash
# Build research/paper/main.pdf on Oscar (no TeX on the Mac) and fetch it; optional page renders for checking.
# usage: bash scripts/paper_build.sh [PAGES_DIR]   (needs the shared ssh connection, see scripts/oscar.sh)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
SSHO="-o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o BatchMode=yes -o LogLevel=ERROR"
H=aagraw73@ssh.ccv.brown.edu
R=/oscar/scratch/aagraw73/speedrun/paper
PD=${1:-}
ssh $SSHO $H "mkdir -p $R/figures"
scp -q $SSHO "$REPO/research/paper/main.tex" "$REPO/research/paper/refs.bib" $H:$R/
scp -q $SSHO "$REPO"/research/paper/figures/*.pdf $H:$R/figures/
ssh $SSHO $H "bash -lc 'cd $R && module load texlive/20240312-xnyd >/dev/null 2>&1 && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1; echo rc=\$?; \
  grep -E \"^!|^l\\.[0-9]|Overfull|undefined\" main.log | sort | uniq -c | sort -rn | head -20; \
  grep \"Output written\" main.log; if [ -n \"$PD\" ]; then rm -rf pages_$PD; mkdir -p pages_$PD && pdftoppm -r 80 -png main.pdf pages_$PD/p; fi'"
scp -q $SSHO "$H:$R/main.pdf" "$REPO/research/paper/main.pdf"
if [ -n "$PD" ]; then mkdir -p "$REPO/../pages/$PD"; scp -q $SSHO "$H:$R/pages_$PD/*.png" "$REPO/../pages/$PD/"; echo "pages in $REPO/../pages/$PD"; fi
