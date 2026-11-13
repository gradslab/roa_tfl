#!/bin/bash
# Dirichlet-Zubov baseline: tanh-scale factors 30 and 100; seeds 1-3 retrained if one of them wins on seed 0.  usage: run_dz2.sh circle|sinusoid|quadrotor
set -e
export OMP_NUM_THREADS=2
case "$1" in circle) N=circ3 ;; sinusoid) N=sin6 ;; quadrotor) N=quad3 ;; *) echo usage; exit 1 ;; esac
C=$1
for f in 30 100; do
  python -u zpinn_dz.py $C 0 runs/${N}_dzf${f}_s0 $f
  python -u certify3.py runs/${N}_dzf${f}_s0 --top 3
done
BEST=$(python3 -I -c "
import json
v={f:json.load(open(f'runs/${N}_dzf{f}_s0/certify_v3.json'))['coverage'] for f in (1,3,10,30,100)}
print(max(v,key=v.get))")
echo "BEST FACTOR $BEST"
OLD=$(python3 -I -c "import json;print(int(json.load(open('runs/${N}_dz_s0/dz.json'))['factor']))")
if [ "$BEST" != "$OLD" ]; then
  rm -rf runs/${N}_dz_s0; cp -r runs/${N}_dzf${BEST}_s0 runs/${N}_dz_s0
  runs=""; for s in 1 2 3; do rm -rf runs/${N}_dz_s$s; python -u zpinn_dz.py $C $s runs/${N}_dz_s$s $BEST; runs="$runs runs/${N}_dz_s$s"; done
  python -u certify3.py $runs --top 3
fi
echo "DZ2 DONE $N"
