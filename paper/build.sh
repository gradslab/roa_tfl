#!/bin/sh
# Build main.pdf; prints LaTeX errors and the counts of overfull boxes and undefined references.
cd "$(dirname "$0")"
latexmk -gg -pdf -bibtex -interaction=nonstopmode main.tex > build.log 2>&1
status=$?
grep -n "^!" -A3 main.log | head -12
echo "overfull: $(grep -c Overfull main.log)  undefined: $(grep -cE '(Citation|Reference).*undefined' main.log)"
exit $status
