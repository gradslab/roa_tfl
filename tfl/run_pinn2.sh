#!/bin/bash
# Zubov PINN baseline: tanh scale chosen on seed 0 over factors 1 to 300, seeds 1-3 retrained with the best factor.  usage: run_pinn2.sh circle|sinusoid|quadrotor
set -e
export OMP_NUM_THREADS=2
case "$1" in circle) N=circ3 ;; sinusoid) N=sin6 ;; quadrotor) N=quad3 ;; *) echo usage; exit 1 ;; esac
C=$1
for k in 0 1 2 3; do [ -d runs/${N}_pinnf1_s$k ] || cp -r runs/${N}_pinn_s$k runs/${N}_pinnf1_s$k; done
for f in 3 10 30 100 300; do
  python -u zpinn_liu.py pinn $C 0 runs/${N}_pinnf${f}_s0 $f
  python -u certify3.py runs/${N}_pinnf${f}_s0 --top 3
done
BEST=$(python3 -I -c "
import json
v={f:json.load(open(f'runs/${N}_pinnf{f}_s0/certify_v3.json'))['coverage'] for f in (1,3,10,30,100,300)}
print(max(v,key=v.get))")
echo "BEST FACTOR $BEST"
if [ "$BEST" != "1" ]; then
  rm -rf runs/${N}_pinn_s0; cp -r runs/${N}_pinnf${BEST}_s0 runs/${N}_pinn_s0
  runs=""; for s in 1 2 3; do rm -rf runs/${N}_pinn_s$s; python -u zpinn_liu.py pinn $C $s runs/${N}_pinn_s$s $BEST; runs="$runs runs/${N}_pinn_s$s"; done
  python -u certify3.py $runs --top 3
fi
echo "$BEST" > runs/${N}_pinn_s0/factor.txt
echo "PINN2 DONE $N"
