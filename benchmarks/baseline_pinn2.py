r"""Zubov PINN of Liu et al. (Automatica 2025) on the equilibrium benchmarks with the scale search of the Dirichlet-Zubov baseline: W = tanh(a V), a = alpha / factor, the factor chosen on seed 0
by coverage over {1, 3, 10, 30, 100, 300}; sublevel sets restricted to the admissible set; same selection and i.i.d. final test.   Run:  python -u baseline_pinn2.py vdp vdp3 pendulum con3d power
Writes results/baseline_pinn2_<name>.json."""
import json
import sys
import time
from pathlib import Path

import torch

import baseline_dz as DZ
import baselines as B
import core
import systems as S

R = Path(__file__).parent / "results"


def run(s, prob, name, seed, factor):
    m = B.train_pinn(s, prob, seed, factor=factor)
    model = DZ.DzClip(m, s)
    c = B.certify_fn(model, prob, f"pinn{factor:g}")
    r = DZ.test(model, prob, f"pinn2_{name}_s{seed}", c["c1"], c["c2"])
    r.update(seed=seed, factor=factor)
    print(f"{name} seed {seed} factor {factor:g}: c1 {r['c1']:.4f} c2 {r['c2']:.4f} coverage {r['coverage']:.2f} FP {r['false_positive']:.3f}", flush=True)
    return r


for name in sys.argv[1:]:
    t0 = time.time()
    s = S.get(name)
    prob = core.Problem(s, core.Config(system=name))
    prob.build_inner(verbose=False)
    grid = [run(s, prob, name, 0, f) for f in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)]
    best = max(grid, key=lambda r: r["coverage"])
    res = [best] + [run(s, prob, name, k, best["factor"]) for k in (1, 2, 3)]
    (R / f"baseline_pinn2_{name}.json").write_text(json.dumps(dict(grid=grid, runs=res), indent=1))
    print(f"{name}: factor {best['factor']:g}  coverage {[round(r['coverage'], 2) for r in res]}  [{time.time() - t0:.0f} s]", flush=True)
