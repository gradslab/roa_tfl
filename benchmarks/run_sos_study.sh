#!/bin/sh
# Sum-of-squares settings study on Van der Pol (mu = 1 and 3) -> results/sos_vdp_study/*.json and results/sos_vdp_study.json.
set -e
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
L=results/sos_vdp_study
mkdir -p $L
PY="python -u sos_sound.py"
LIU="--shape id --init x --s1deg 2 --s2deg 4 --box 0 --iters 30 --stall 0"
for sysn in vdp3 vdp; do
  $PY $sysn --samples 0 --out $L/mu_${sysn}_ours.json
  $PY $sysn $LIU --samples 0 --out $L/mu_${sysn}_liu.json
  $PY $sysn --shape id --init x --s1deg 2 --s2deg 4 --box 0 --iters 100 --stall 0 --samples 0 --out $L/mu_${sysn}_liu_it100.json
  $PY $sysn $LIU --lenient 1 --samples 0 --out $L/mu_${sysn}_liu_lenient.json
done
$PY vdp3 --shape id --init x --s1deg 2 --s2deg 4 --box 1 --iters 30 --stall 0 --samples 0 --out $L/mu_vdp3_liu_box.json
$PY vdp3 --shape id --samples 0 --out $L/mu_vdp3_ours_shapeid.json
$PY vdp3 --init x --samples 0 --out $L/mu_vdp3_ours_initx.json
$PY vdp3 --s1deg 2 --samples 0 --out $L/mu_vdp3_ours_s1deg2.json
$PY vdp3 --box 0 --samples 0 --out $L/mu_vdp3_ours_nobox.json
$PY vdp3 --shape id --init x --s1deg 2 --s2deg 4 --box 0 --samples 0 --out $L/mu_vdp3_liu_stall.json
$PY vdp3 --shape id --init x --box 0 --iters 30 --stall 0 --samples 0 --out $L/mu_vdp3_liu_ourdeg.json
$PY vdp3 --shape lin --init x --s1deg 2 --s2deg 4 --box 0 --iters 30 --stall 0 --samples 0 --out $L/mu_vdp3_liu_shapelin.json
$PY vdp3 --shape id --init y --s1deg 2 --s2deg 4 --box 0 --iters 30 --stall 0 --samples 0 --out $L/mu_vdp3_liu_inity.json
$PY vdp3 --shape id --init x --s1deg 4 --s2deg 4 --box 0 --iters 30 --stall 0 --samples 0 --out $L/mu_vdp3_liu_s1deg4.json
python sos_vdp_study.py $L/mu_*.json
