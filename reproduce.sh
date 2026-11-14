#!/bin/bash
# Reproduce the numbers, tables, figure and PDF of the paper.  usage: ./reproduce.sh setup|tables|figure|paper|tfl|benchmarks|all
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
GEN="results3.tex table_main3.tex table_seeds3.tex table_consts3.tex table_inner3.tex table_ablation3.tex table_step3.tex"
B="vdp vdp3 pendulum con3d power"

setup() {
  python -m pip install -r "$ROOT/requirements.txt"
}

tables() {
  (cd "$ROOT/tfl" && python make_results3.py) || { echo "ledger failed"; return 1; }
  local bad=0
  for f in $GEN; do
    if ! diff -u "$ROOT/paper/reference/$f" "$ROOT/paper/$f"; then bad=1; fi
  done
  if [ $bad = 0 ]; then echo "OK: every generated file matches paper/reference/"; else echo "DIFFERENCES found (see above)"; return 1; fi
}

figure() {
  (cd "$ROOT/tfl" && python figs3.py)
}

paper() {
  sh "$ROOT/paper/build.sh"
}

fresh() {
  local dir=$1 d; shift
  for d in "$@"; do
    if [ -e "$dir/$d.stored" ]; then echo "$dir/$d.stored exists; move it away first"; exit 1; fi
  done
  for d in "$@"; do
    [ -e "$dir/$d" ] && mv "$dir/$d" "$dir/$d.stored"
    mkdir -p "$dir/$d"
  done
}

tfl() {
  cd "$ROOT/tfl"
  fresh "$ROOT/tfl" runs results runs_short_seed0
  cp results.stored/inner_contact.json results/
  python -u inner_region.py --system unicycle
  python -u inner_region.py --system unicycle --path "y-sin(x)" --pi x --periodic-x1 true
  python -u inner_region.py --system quadrotor
  for c in circle sinusoid quadrotor; do
    bash run_stream.sh $c
    bash run_pinn2.sh $c
    bash run_dz.sh $c
    bash run_dz2.sh $c
    bash run_rich.sh $c
  done
  python -u oracle_tfl.py
  python -u ref_sensitivity.py
  python -u floor3.py
  python -u unsafe3.py circ3 sin6 quad3
  python -u unsafe_check3.py circ3 sin6 quad3
}

benchmarks() {
  cd "$ROOT/benchmarks"
  fresh "$ROOT/benchmarks" runs results runs_long_seed0
  python -u gt.py $B
  sh run_bench.sh
}

case "$1" in
  setup) setup ;;
  tables) tables ;;
  figure) figure ;;
  paper) paper ;;
  tfl) tfl ;;
  benchmarks) benchmarks ;;
  all) (benchmarks); (tfl); figure; tables || true; paper ;;
  *) echo "usage: $0 setup|tables|figure|paper|tfl|benchmarks|all"; exit 1 ;;
esac
