#!/bin/bash
# Seed-0 configuration runs of Appendix G (final test on 2e5 samples), then the seed-0 runs of run_stream.sh as copies of the kept ones.
set -e
export OMP_NUM_THREADS=2
G=(--beta 0.1 --h-pow 2 --ema 0.999 --epochs 100 --snap-every 5 --step-tau 1.0 --step-sub 40 --cex-step true --mu-step 100 --step-margin 0.2 --cex-every 5 --cex-buffer 100000 --cex-rolling true --cex-worst true --kappa 0.3 --w-cap 3 --v-cap 8 --device cpu)
probe() {
  case "$1" in
    circle)    ARGS=() ;;
    sinusoid)  ARGS=(--path "y-sin(x)" --pi x --periodic-x1 true) ;;
    quadrotor) ARGS=(--system quadrotor) ;;
  esac
  python -u zpinn.py "${G[@]}" "${ARGS[@]}" "${@:3}" --seed 0 --out runs/cfg$2_$1
  python -u certify3.py runs/cfg$2_$1 --top 3 --tag _probe --n-iid 200000
}
probe circle O
probe circle M10W1 --mu-step 10 --w-cap 1
probe sinusoid O
probe sinusoid M10W1 --mu-step 10 --w-cap 1
probe quadrotor O
probe quadrotor M0 --mu-step 0 --cex-every 0 --step-tau 0
probe quadrotor M10 --mu-step 10
probe quadrotor W1 --w-cap 1
probe quadrotor M10W1 --mu-step 10 --w-cap 1
probe quadrotor M3W1 --mu-step 3 --w-cap 1
probe quadrotor M30W1 --mu-step 30 --w-cap 1
probe quadrotor M10W0 --mu-step 10 --w-cap 0
probe quadrotor M10W1H128 --mu-step 10 --w-cap 1 --hidden 128 128 128
for p in "cfgO_circle circ3_s0" "cfgO_sinusoid sin6_s0" "cfgM10W0_quadrotor quad3_s0"; do
  set -- $p
  [ -d runs/$2 ] || cp -r runs/$1 runs/$2
done
