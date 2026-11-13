#!/bin/bash
# Dirichlet-Zubov PINN baseline (Meng & Liu 2026) on one TFL case: tanh scale chosen on seed 0 by coverage, then seeds 1-3.  usage: run_dz.sh circle|sinusoid|quadrotor
set -e
export OMP_NUM_THREADS=2
case "$1" in circle) N=circ3 ;; sinusoid) N=sin6 ;; quadrotor) N=quad3 ;; *) echo usage; exit 1 ;; esac
C=$1
for f in 1 3 10; do
  python -u zpinn_dz.py $C 0 runs/${N}_dzf${f}_s0 $f
  python -u certify3.py runs/${N}_dzf${f}_s0 --top 3
done
BEST=$(python3 -I -c "
import json
v={f:json.load(open(f'runs/${N}_dzf{f}_s0/certify_v3.json'))['coverage'] for f in (1,3,10)}
print(max(v,key=v.get))")
echo "BEST FACTOR $BEST"
rm -rf runs/${N}_dz_s0; cp -r runs/${N}_dzf${BEST}_s0 runs/${N}_dz_s0
runs=""; for s in 1 2 3; do python -u zpinn_dz.py $C $s runs/${N}_dz_s$s $BEST; runs="$runs runs/${N}_dz_s$s"; done
python -u certify3.py $runs --top 3
echo "DZ DONE $N"
