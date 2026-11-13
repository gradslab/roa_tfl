#!/bin/bash
# Reproduce the numbers, tables, figure and PDF of the paper.  usage: ./reproduce.sh setup|tables|figure|paper|tfl|benchmarks|all
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
GEN="results3.tex table_main3.tex table_seeds3.tex table_consts3.tex table_inner3.tex table_ablation3.tex table_bench3.tex table_core3.tex"
B="vdp vdp3 pendulum con3d power"
O="vdp_obs1 vdp_obs2 power_obs2 power_obs4"

setup() {
  python -m pip install -r "$ROOT/requirements.txt"
}

tables() {
  (cd "$ROOT/tfl" && python make_results3.py)
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
  for d in runs results; do
    if [ -e "$1/$d.stored" ]; then echo "$1/$d.stored exists; move it away first"; exit 1; fi
    [ -e "$1/$d" ] && mv "$1/$d" "$1/$d.stored"
    mkdir -p "$1/$d"
  done
}

tfl() {
  cd "$ROOT/tfl"
  fresh "$ROOT/tfl"
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
  python -u ref_sensitivity.py
  python -u floor3.py
  python -u unsafe3.py circ3 sin6 quad3
  python -u unsafe_check3.py circ3 sin6 quad3
}

benchmarks() {
  cd "$ROOT/benchmarks"
  fresh "$ROOT/benchmarks"
  python -u gt.py $B $O
  sh run_bench.sh
  sh run_obs.sh vdp_obs1 vdp_obs2
  sh run_obs.sh power_obs2 power_obs4
  python obs_summary.py
  python -u baselines.py quad $B
  python -u baselines.py pinn $B
  python -u certify_iid.py $B
  python -u baseline_pinn2.py $B
  for sys in $B; do
    DZ_EXTRA=1,3,10,30,100 python -u baseline_dz.py $sys
    best=$(python -c "import json; d=json.load(open('results/baseline_dz_$sys.json')); print(int(max(d['grid'], key=lambda r: r['coverage'])['factor']))")
    if [ "$best" -ge 100 ]; then DZ_EXTRA=300,1000 python -u baseline_dz.py $sys; fi
  done
  python -u core_proof.py $B $O
  for k in ours quad pinn pinn2 dz; do python -u certify_core.py $k $B; done
  for k in pinn pinn2 dz; do python -u certify_core.py --noclamp $k $B; done
  python -u certify_core.py summary $B
  for k in ours quad pinn; do python -u certify_core.py $k vdp_obs1 power_obs2; done
  python -u certify_core.py --noclamp pinn vdp_obs1 power_obs2
  python -u certify_core.py summary vdp_obs1 power_obs2
  python -u core_q_check.py $B $O
  for k in ours quad pinn pinn2 dz; do python -u certify_core.py --relaxed $k $B $O; done
  python -u certify_core.py --relaxed summary $B $O
  python -u unsafe_bench.py --core $B vdp_obs1 power_obs2
  python -u unsafe_bench_check.py --core $B vdp_obs1 power_obs2
  python -u unsafe_bench.py --core --relaxed $B $O
  python -u unsafe_bench_check.py --core --relaxed $B $O
  python box_effect.py $O
  python -u sos.py vdp vdp3 con3d
  python -u sos_sound.py pendulum power
  for s in vdp vdp3 con3d; do
    python -u sos_sound.py $s --fixedV results/baseline_sos_$s.json --out results/sos_recert_$s.json
  done
  python sos_settings.py
  sh run_sos_study.sh
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
