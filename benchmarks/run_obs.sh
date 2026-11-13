#!/bin/sh
# Obstacle benchmarks of Meng and Liu: the pipeline of run_bench.sh, then the baselines.  Steps with results are skipped.
# Run after python -u gt.py <systems>.  usage: sh run_obs.sh vdp_obs1 vdp_obs2
set -e
for sys in "$@"; do
  for s in 0 1 2 3; do
    [ -f runs/${sys}_s$s/snapshots/epoch_100.pt ] || python -u core.py --system $sys --seed $s --out runs/${sys}_s$s
  done
  for s in 0 1 2 3; do [ -f runs/${sys}_s$s/certify.json ] || python -u certify.py runs/${sys}_s$s; done
  python -u union.py runs/${sys}_s0 runs/${sys}_s1 runs/${sys}_s2 runs/${sys}_s3
  for s in 0 1 2 3; do [ -f runs/${sys}_s${s}_rep/repair.json ] || python -u repair.py runs/${sys}_s$s; done
  for s in 0 1 2 3; do [ -f runs/${sys}_s${s}_rep/certify.json ] || python -u certify.py runs/${sys}_s${s}_rep; done
  python -u union.py runs/${sys}_s0_rep runs/${sys}_s1_rep runs/${sys}_s2_rep runs/${sys}_s3_rep
  echo "MAIN DONE $sys"
  [ -f results/baseline_quad_$sys.json ] || python -u baselines.py quad $sys
  [ -f results/baseline_pinn_$sys.json ] || python -u baselines.py pinn $sys
  [ -f results/iid_union_$sys.json ] || python -u certify_iid.py $sys
  [ -f results/baseline_pinn2_$sys.json ] || python -u baseline_pinn2.py $sys
  [ -f results/baseline_dz_$sys.json ] || DZ_EXTRA=1,3,10,30,100 python -u baseline_dz.py $sys
  best=$(python -c "import json; d=json.load(open('results/baseline_dz_$sys.json')); print(int(max(d['grid'], key=lambda r: r['coverage'])['factor']))")
  if [ "$best" -ge 100 ]; then DZ_EXTRA=300,1000 python -u baseline_dz.py $sys; fi
  echo "OBS DONE $sys"
done
echo "ALL DONE"
