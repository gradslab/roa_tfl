#!/bin/bash
# Sensitivity of the PINN baselines to the input features and network size (seed 0, candidate features, 64-64-64).  usage: run_rich.sh circle|sinusoid|quadrotor
set -e
export OMP_NUM_THREADS=2 PINN_RICH=1
case "$1" in circle) N=circ3 ;; sinusoid) N=sin6 ;; quadrotor) N=quad3 ;; *) echo usage; exit 1 ;; esac
C=$1
for f in 1 10; do
  python -u zpinn_liu.py pinn $C 0 runs/${N}_richpinn_f${f}_s0 $f
  python -u certify3.py runs/${N}_richpinn_f${f}_s0 --top 3
  python -u zpinn_dz.py $C 0 runs/${N}_richdz_f${f}_s0 $f
  python -u certify3.py runs/${N}_richdz_f${f}_s0 --top 3
done
echo "RICH DONE $N"
