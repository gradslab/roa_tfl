#!/bin/sh
# Five benchmarks: 4 seeds, level selection, union, fine-tuning, union.  Steps with results are skipped.  usage: sh run_bench.sh
set -e
for sys in vdp pendulum con3d power vdp3; do
  for s in 0 1 2 3; do
    [ -f runs/${sys}_s$s/snapshots/epoch_100.pt ] || python -u core.py --system $sys --seed $s --out runs/${sys}_s$s
  done
  for s in 0 1 2 3; do [ -f runs/${sys}_s$s/certify.json ] || python -u certify.py runs/${sys}_s$s; done
  python -u union.py runs/${sys}_s0 runs/${sys}_s1 runs/${sys}_s2 runs/${sys}_s3
  for s in 0 1 2 3; do [ -f runs/${sys}_s${s}_rep/repair.json ] || python -u repair.py runs/${sys}_s$s; done
  for s in 0 1 2 3; do [ -f runs/${sys}_s${s}_rep/certify.json ] || python -u certify.py runs/${sys}_s${s}_rep; done
  python -u union.py runs/${sys}_s0_rep runs/${sys}_s1_rep runs/${sys}_s2_rep runs/${sys}_s3_rep
done
echo "ALL DONE"
