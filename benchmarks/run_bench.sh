#!/bin/sh
# Seed 0 of the five benchmarks with both step settings (long step into runs_long_seed0/, short step = the defaults into
# runs/): training, level selection, fine-tuning, level selection of the fine-tuned run.  Then the step setting of each
# system (step_rule.py).  Steps with results are skipped.  usage: sh run_bench.sh
set -e
B="vdp vdp3 pendulum con3d power"
one() {
  run=$1/${2}_s0; name=$2; shift 2
  [ -f $run/snapshots/epoch_100.pt ] || python -u core.py --system $name --seed 0 "$@" --out $run
  [ -f $run/certify.json ] || python -u certify.py $run
  [ -f ${run}_rep/repair.json ] || python -u repair.py $run
  [ -f ${run}_rep/certify.json ] || python -u certify.py ${run}_rep
}
for sys in $B; do
  one runs_long_seed0 $sys --tau 2 --sub 80 --gamma-num 1 --w-exit 10
  one runs $sys
done
python -u step_rule.py $B
echo "ALL DONE"
