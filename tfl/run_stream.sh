#!/bin/bash
# Full pipeline of one TFL case: baselines, the short-step seed-0 run of the step setting, 4 long-step seeds, level selection,
# fine-tuning, union, exact v, no-decomposition baseline, ablations.  usage: run_stream.sh circle|sinusoid|quadrotor
set -e
export OMP_NUM_THREADS=3
G=(--beta 0.1 --h-pow 2 --ema 0.999 --epochs 100 --snap-every 5 --step-tau 1.0 --step-sub 40 --cex-step true --step-margin 0.2 --cex-every 5 --cex-buffer 100000 --cex-rolling true --cex-worst true --v-cap 8 --kappa 0.3 --device cpu)
LONG=(--tau 2.0 --sub 80 --gamma-num 1 --w-exit 10)
SEEDS="0 1 2 3"
case "$1" in
  circle)    NAME=circ3; ARGS=(--mu-step 100 --w-cap 3) ;;
  sinusoid)  NAME=sin6;  ARGS=(--mu-step 100 --w-cap 3 --path "y-sin(x)" --pi x --periodic-x1 true) ;;
  quadrotor) NAME=quad3; ARGS=(--mu-step 10 --w-cap 0 --system quadrotor) ;;
  *) echo usage; exit 1 ;;
esac
CASE=$1
python -u zpinn_liu.py quad $CASE 0 runs/${NAME}_quadratic
python -u certify3.py runs/${NAME}_quadratic --top 1
runs=""; for s in $SEEDS; do python -u zpinn_liu.py pinn $CASE $s runs/${NAME}_pinn_s$s; runs="$runs runs/${NAME}_pinn_s$s"; done
python -u certify3.py $runs --top 3
echo "BASELINES DONE $NAME"
# step setting: seed 0 with the short step (the defaults); the long step is kept on the three cases, which the ledger checks
mkdir -p runs_short_seed0
[ -f runs_short_seed0/${NAME}_s0/model.pt ] || python -u zpinn.py "${G[@]}" "${ARGS[@]}" --seed 0 --out runs_short_seed0/${NAME}_s0
python -u certify3.py runs_short_seed0/${NAME}_s0 --top 3
echo "SHORT STEP DONE $NAME"
post() {
  local n=$1 runs="" rep=""
  for s in $SEEDS; do runs="$runs runs/${n}_s$s"; rep="$rep runs/${n}_s${s}_rep3"; done
  python -u certify3.py $runs --top 3
  python -u repair3.py $runs
  python -u certify3.py $rep --top 3
  python -u carry_level3.py $n
  python -u union3.py $n
}
for s in $SEEDS; do [ -f runs/${NAME}_s$s/model.pt ] || python -u zpinn.py "${G[@]}" "${LONG[@]}" "${ARGS[@]}" --seed $s --out runs/${NAME}_s$s; done
post $NAME
python -u ceiling3.py $NAME
echo "MAIN DONE $NAME"
for s in $SEEDS; do python -u zpinn_whole.py "${G[@]}" "${LONG[@]}" "${ARGS[@]}" --seed $s --out runs/${NAME}w_s$s; done
post ${NAME}w
echo "WHOLE DONE $NAME"
abl() {
  local key=$1; shift
  python -u zpinn.py "${G[@]}" "${LONG[@]}" "${ARGS[@]}" --seed 0 "$@" --out runs/${NAME}_abl_$key
  python -u certify3.py runs/${NAME}_abl_$key --top 3
}
if [ "$CASE" = circle ]; then
  abl pinn_style --loss residual
  abl no_step --mu-step 0 --cex-every 0 --step-tau 0
  abl no_wcap --w-cap 0
  abl sum_ansatz --ansatz sum
fi
ZW_NO_COST=1 python -u zpinn_whole.py "${G[@]}" "${LONG[@]}" "${ARGS[@]}" --seed 0 --out runs/${NAME}_abl_safety_value
python -u certify3.py runs/${NAME}_abl_safety_value --top 3
python -u ablation_summary3.py
echo "STREAM DONE $NAME"
